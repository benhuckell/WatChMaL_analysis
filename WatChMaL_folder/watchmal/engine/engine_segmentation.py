# torch imports
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
#from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

# hydra imports
from hydra.utils import instantiate

# generic imports
from math import floor, ceil
import numpy as np
from numpy import savez
import os
from time import strftime, localtime, time
import sys
from sys import stdout
import copy
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# WatChMaL imports
from watchmal.dataset.data_utils import get_segmentation_data_loader
from watchmal.utils.logging_utils import CSVData

#extraneous testing imports

class ClassifierEngine:
    def __init__(self, model, rank, gpu, dump_path):
        # create the directory for saving the log and dump files
        self.dirpath = dump_path

        self.rank = rank

        self.model = model

        self.device = torch.device(gpu)

        # Setup the parameters to save given the model type
        if isinstance(self.model, DDP):
            self.is_distributed = True
            self.model_accs = self.model.module
            self.ngpus = torch.distributed.get_world_size()
        else:
            self.is_distributed = False
            self.model_accs = self.model

        self.data_loaders = {}

        self.mpmt_positions = None

        self.criterion = nn.CrossEntropyLoss(reduction = "none")
        self.softmax = nn.Softmax(dim=1)

        # define the placeholder attributes
        self.data      = None
        self.labels    = None
        self.energies  = None
        self.eventids  = None
        self.rootfiles = None
        self.angles    = None
        self.event_ids = None
        
        # logging attributes
        self.train_log = CSVData(self.dirpath + "log_train_{}.csv".format(self.rank))

        if self.rank == 0:
            self.val_log = CSVData(self.dirpath + "log_val.csv")
    
    def configure_optimizers(self, optimizer_config):
        """
        Set up optimizers from optimizer config
        """
        self.optimizer = instantiate(optimizer_config, params=self.model_accs.parameters())

    def configure_data_loaders(self, data_config, loaders_config, is_distributed, seed):
        """
        Set up data loaders from loaders config
        """
        for name, loader_config in loaders_config.items():
            self.data_loaders[name], self.mpmt_positions = get_segmentation_data_loader(**data_config, **loader_config, is_distributed=is_distributed, seed=seed)
    
    def get_synchronized_metrics(self, metric_dict):
        global_metric_dict = {}
        for name, array in zip(metric_dict.keys(), metric_dict.values()):
            tensor = torch.as_tensor(array).to(self.device)
            global_tensor = [torch.zeros_like(tensor).to(self.device) for i in range(self.ngpus)]
            torch.distributed.all_gather(global_tensor, tensor)
            global_metric_dict[name] = torch.cat(global_tensor)
        
        return global_metric_dict

    def forward(self, train=True):
        """
        Compute predictions and metrics for a batch of data.

        Parameters:
            train = whether to compute gradients for backpropagation
            self should have attributes model, criterion, softmax, data, label
        
        Returns : a dict of loss, predicted labels, softmax, accuracy, and raw model outputs
        """

        with torch.set_grad_enabled(train):
            # Move the data and the labels to the GPU (if using CPU this has no effect)
            self.data = self.data.to(self.device) #shape (batch_size, 19, 40, 40)
            self.labels = self.labels.to(self.device)

            #print("Model Input size:", self.data.shape)
            #print("Labels Tensor size:", self.labels.shape)

            model_out = self.model(self.data) #predictions are generated, shape (batch_size, # classes, 40, 40)
            

            #print("Model Output size:", model_out.shape)
            #print(model_out[0,:,0,0])
            #print("Labels Tensor size:", self.labels.shape)
            #print(torch.count_nonzero(self.labels))
            #print(torch.unique(self.labels))

            #Calculate first loss
            regLoss = torch.sum(self.criterion(model_out.float(), self.labels), dim=[1,2,3])
            #print(type(regLoss), type(test))

            #Calculate swapped loss
            swapLabels = copy.deepcopy(self.labels)
            swapLabels[swapLabels==2] = -3
            swapLabels[swapLabels==3] = 2
            swapLabels[swapLabels==-3] = 3
            swapLoss = torch.sum(self.criterion(model_out.float(), swapLabels), dim=[1,2,3])

            softmax          = self.softmax(model_out)
            predicted_labels = torch.argmax(model_out,dim=1)

            self.loss = torch.mean(torch.min(regLoss, swapLoss)) #Calculate the overall loss as the mean of the tensor of minimums from the two loss methods.
            #print(type(self.loss))
            lossPositions = torch.gt(regLoss, swapLoss) # Element-wise true if should use swapLoss, false if should use regLoss

            '''
            correctlyIdentified = ((predicted_labels == self.labels) & (self.labels != 0)).sum().item()
            total = (self.labels != 0).sum().item()
            accuracy = correctlyIdentified/total
            '''

            accArray = []
            for eventNum in range(self.labels.shape[0]):
                if(lossPositions[eventNum]): #regular loss > swap loss -> use swap
                    correct = ((swapLabels[eventNum]  == predicted_labels[eventNum] ) & (swapLabels[eventNum]  != 0)).sum().item()
                else: #regular loss <= swap loss -> use regular
                    correct = ((self.labels[eventNum] == predicted_labels[eventNum] ) & (self.labels[eventNum]  != 0)).sum().item()

                total = (self.labels[eventNum]  != 0).sum().item()
                accArray.append(correct/total)

            accuracy = np.mean(accArray)
    
        
        return {'loss'             : self.loss.detach().cpu().item(),
                'predicted_labels' : predicted_labels.detach().cpu().numpy(),
                'softmax'          : softmax.detach().cpu().numpy(),
                'accuracy'         : accuracy,
                'raw_pred_labels'  : model_out}

    def backward(self):
        self.optimizer.zero_grad()  # reset accumulated gradient
        self.loss.backward()        # compute new gradient
        self.optimizer.step()       # step params
    
    # ========================================================================

    def train(self, train_config):
        """
        Train the model on the training set.
        
        Parameters : None
        
        Outputs :
            total_val_loss = accumulated validation loss
            avg_val_loss = average validation loss
            total_val_acc = accumulated validation accuracy
            avg_val_acc = accumulated validation accuracy
            
        Returns : None
        """

        # initialize training params
        epochs          = train_config.epochs
        report_interval = train_config.report_interval
        val_interval    = train_config.val_interval
        num_val_batches = train_config.num_val_batches
        checkpointing   = train_config.checkpointing

        # set the iterations at which to dump the events and their metrics
        if self.rank == 0:
            print(f"Training... Validation Interval: {val_interval}")

        # set model to training mode
        self.model.train()

        # initialize epoch and iteration counters
        epoch = 0.
        self.iteration = 0

        # keep track of the validation accuracy
        best_val_acc = 0.0
        best_val_loss = 1.0e6

        # initialize the iterator over the validation set
        val_iter = iter(self.data_loaders["validation"])

        # global training loop for multiple epochs
        while (floor(epoch) < epochs):
            if self.rank == 0:
                print('Epoch',floor(epoch), 'Starting @', strftime("%Y-%m-%d %H:%M:%S", localtime()))
            
            times = []

            start_time = time()

            train_loader = self.data_loaders["train"]

            # update seeding for distributed samplers
            if self.is_distributed:
                train_loader.sampler.set_epoch(epoch)

            print("Outside train loop")

            # local training loop for batches in a single epoch
            for i, train_data in enumerate(self.data_loaders["train"]):

                #print("Event IDs:",i,train_data["event_ids"].shape)
                #print("Labels:", i, train_data["segmentation"].shape)
                #print("Data:", i, train_data["data"].shape)
                
                # run validation on given intervals
                if self.iteration % val_interval == 2:
                    # set model to eval mode
                    self.model.eval()

                    val_metrics = {"iteration": self.iteration, "epoch": epoch, "loss": 0., "accuracy": 0., "saved_best": 0}

                    for val_batch in range(num_val_batches):
                        try:
                            val_data = next(val_iter)
                        except StopIteration:
                            del val_iter
                            val_iter = iter(self.data_loaders["validation"])
                            val_data = next(val_iter)
                        
                        # extract the event data from the input data tuple
                        self.data      = val_data['data'].float()
                        self.data = torch.unsqueeze(self.data,1)
                        self.labels    = val_data['segmented_labels'].long()
                        self.energies  = val_data['energies'].float()
                        self.angles    = val_data['angles'].float()
                        self.event_ids = val_data['event_ids'].float()

                        val_res = self.forward(False)
                        
                        val_metrics["loss"] += val_res["loss"]
                        val_metrics["accuracy"] += val_res["accuracy"]
                    
                    # return model to training mode
                    self.model.train()

                    # record the validation stats to the csv
                    val_metrics["loss"] /= num_val_batches
                    val_metrics["accuracy"] /= num_val_batches

                    local_val_metrics = {"loss": np.array([val_metrics["loss"]]), "accuracy": np.array([val_metrics["accuracy"]])}

                    if self.is_distributed:
                        global_val_metrics = self.get_synchronized_metrics(local_val_metrics)
                        for name, tensor in zip(global_val_metrics.keys(), global_val_metrics.values()):
                            global_val_metrics[name] = np.array(tensor.cpu())
                    else:
                        global_val_metrics = local_val_metrics

                    if self.rank == 0:
                        # Save if this is the best model so far
                        global_val_loss = np.mean(global_val_metrics["loss"])
                        global_val_accuracy = np.mean(global_val_metrics["accuracy"])

                        val_metrics["loss"] = global_val_loss
                        val_metrics["accuracy"] = global_val_accuracy

                        if val_metrics["loss"] < best_val_loss:
                            print('best validation loss so far!: {}'.format(best_val_loss))
                            self.save_state(best=True)
                            val_metrics["saved_best"] = 1

                            best_val_loss = val_metrics["loss"]

                        # Save the latest model if checkpointing
                        if checkpointing:
                            self.save_state(best=False)
                                        
                        self.val_log.record(val_metrics)
                        self.val_log.write()
                        self.val_log.flush()
                
                # Train on batch
                self.data      = train_data['data'].float()
                self.data = torch.unsqueeze(self.data,1)
                self.labels    = train_data['segmented_labels'].long()
                self.energies  = train_data['energies'].float()
                self.angles    = train_data['angles'].float()
                self.event_ids = train_data['event_ids'].float()

                # Call forward: make a prediction & measure the average error using data = self.data
                res = self.forward(True)

                #Call backward: backpropagate error and update weights using loss = self.loss
                self.backward()

                # update the epoch and iteration
                #print(self.data_loaders["train"])
                epoch          += 1./len(self.data_loaders["train"])
                self.iteration += 1
                
                # get relevant attributes of result for logging
                train_metrics = {"iteration": self.iteration, "epoch": epoch, "loss": res["loss"], "accuracy": res["accuracy"]}
                
                # record the metrics for the mini-batch in the log
                self.train_log.record(train_metrics)
                self.train_log.write()
                self.train_log.flush()
                
                
                # print the metrics at given intervals
                if self.rank == 0 and self.iteration % report_interval == 0:
                    print("... Iteration %d ... Epoch %1.2f ... Training Loss %1.3f ... Training Accuracy %1.3f" %
                          (self.iteration, epoch, res["loss"], res["accuracy"]))
                
                if epoch >= epochs:
                    break
        
        self.train_log.close()
        if self.rank == 0:
            self.val_log.close()

    def evaluate(self, test_config):
        """
        Evaluate the performance of the trained model on the validation set.
        
        Parameters : None
        
        Outputs : 
            total_val_loss = accumulated validation loss
            avg_val_loss = average validation loss
            total_val_acc = accumulated validation accuracy
            avg_val_acc = accumulated validation accuracy
            
        Returns : None
        """
        print("evaluating in directory: ", self.dirpath)
        
        # Variables to output at the end
        eval_loss = 0.0
        eval_acc = 0.0
        eval_iterations = 0
        
        # Iterate over the validation set to calculate val_loss and val_acc
        with torch.no_grad():
            
            # Set the model to evaluation mode
            self.model.eval()
            
            # Variables for the confusion matrix
            loss, accuracy, indices, labels, predictions, softmaxes= [],[],[],[],[],[]
            
            # Extract the event data and label from the DataLoader iterator
            
            for it, eval_data in enumerate(self.data_loaders["test"]):

                if(it >= 1):
                    break

                # TODO: see if copying helps
                self.data = copy.deepcopy(eval_data['data'].float())
                self.data = torch.unsqueeze(self.data,1)
                self.labels = copy.deepcopy(eval_data['segmented_labels'].long())
                
                eval_indices = copy.deepcopy(eval_data['indices'].long().to("cpu"))

                # Run the forward procedure and output the result
                result = self.forward(False)

                eval_loss += result['loss']
                eval_acc += result['accuracy']
                
                # Copy the tensors back to the CPU
                self.labels = self.labels.to("cpu")

                #Plot first eventent
                eventNumberToPlot = 75 #Change this to select event, must be in range [0, test_batch_size]
                self.plot_event(eval_data["data"][eventNumberToPlot], self.mpmt_positions, save_file_name = "data.png", cmap=plt.cm.gist_heat_r)
                self.plot_event(eval_data["segmented_labels"][eventNumberToPlot], self.mpmt_positions, save_file_name = "labels.png", cmap=ListedColormap(["white", "gray", "yellow", "green", "red", "blue"]))
                self.plot_event(result["predicted_labels"][eventNumberToPlot], self.mpmt_positions, save_file_name = "predictions.png", cmap=ListedColormap(["white", "gray", "yellow", "green", "red", "blue"]))
                
                # Add the local result to the final result
                indices.extend(eval_indices)
                labels.extend(self.labels)
                predictions.extend(result['predicted_labels'])
                softmaxes.extend(result["softmax"])

                print("eval_iteration : " + str(it) + " eval_loss : " + str(result["loss"]) + " eval_accuracy : " + str(result["accuracy"]))

                eval_iterations += 1
        
        # convert arrays to torch tensors
        print("loss : " + str(eval_loss/eval_iterations) + " accuracy : " + str(eval_acc/eval_iterations))

        iterations = np.array([eval_iterations])
        loss = np.array([eval_loss])
        accuracy = np.array([eval_acc])

        local_eval_metrics_dict = {"eval_iterations":iterations, "eval_loss":loss, "eval_acc":accuracy}
        
        #These are lists of np arrays
        indices     = np.array(indices)
        labels      = np.stack(labels,axis=0)
        predictions = np.stack(predictions,axis=0)
        softmaxes   = np.stack(softmaxes,axis=0)
        
        local_eval_results_dict = {"indices":indices, "labels":labels, "predictions":predictions, "softmaxes":softmaxes}

        if self.is_distributed:
            # Gather results from all processes
            global_eval_metrics_dict = self.get_synchronized_metrics(local_eval_metrics_dict)
            global_eval_results_dict = self.get_synchronized_metrics(local_eval_results_dict)
            
            if self.rank == 0:
                for name, tensor in zip(global_eval_metrics_dict.keys(), global_eval_metrics_dict.values()):
                    local_eval_metrics_dict[name] = np.array(tensor.cpu())
                
                #Will have to adjust these if we ever go to distributed
                indices     = np.array(global_eval_results_dict["indices"].cpu())
                labels      = np.array(global_eval_results_dict["labels"].cpu())
                predictions = np.array(global_eval_results_dict["predictions"].cpu())
                softmaxes   = np.array(global_eval_results_dict["softmaxes"].cpu())
        
        if self.rank == 0:
            print("Sorting Outputs...")
            sorted_indices = np.argsort(indices)

            # Save overall evaluation results
            print("Saving Data...")
            np.save(self.dirpath + "indices.npy", sorted_indices)
            np.save(self.dirpath + "labels.npy", labels[sorted_indices])
            np.save(self.dirpath + "predictions.npy", predictions[sorted_indices])
            np.save(self.dirpath + "softmax.npy", softmaxes[sorted_indices])

            # Compute overall evaluation metrics
            val_iterations = np.sum(local_eval_metrics_dict["eval_iterations"])
            val_loss = np.sum(local_eval_metrics_dict["eval_loss"])
            val_acc = np.sum(local_eval_metrics_dict["eval_acc"])

            print("\nAvg eval loss : " + str(val_loss/val_iterations),
                  "\nAvg eval acc : "  + str(val_acc/val_iterations))




    def channel_to_position(self, channel):
        channel = channel % 19 
        theta = (channel<12)*2*np.pi*channel/12 + ((channel >= 12) & (channel<18))*2*np.pi*(channel-12)/6
        radius = 0.2*(channel<18)+0.2*(channel<12)
        position = [radius*np.cos(theta), radius*np.sin(theta)] # note this is [y, x] or [row, column]
        return position


    def plot_event(self, data, mpmt_pos, save_file_name = "output.png", old_convention=False, **plot_args):

        fig = plt.figure(figsize=(20,12))
        ax = fig.add_subplot(111)
        mpmts = ax.scatter(mpmt_pos[:, 1], mpmt_pos[:, 0], s=380, facecolors='none', edgecolors='0.9')
        indices = np.indices(data.shape)
        channels = indices[0].flatten()
        positions = indices[1:].reshape(2,-1).astype(np.float64)
        positions += self.channel_to_position(channels)
        if old_convention:
            positions[1] = max(mpmt_pos[:, 1])-positions[1]
        pmts = ax.scatter(positions[1], positions[0], c=data.flatten(), s=3, **plot_args)
        plt.colorbar(pmts)

        plt.savefig(save_file_name)
        print("Saved figure as:", save_file_name)

    # ========================================================================
    def restore_best_state(self):
        best_validation_path = "{}{}{}{}".format(self.dirpath,
                                     str(self.model._get_name()),
                                     "BEST",
                                     ".pth")

        self.restore_state_from_file(best_validation_path)
    
    def restore_state(self, restore_config):
        self.restore_state_from_file(restore_config.weight_file)

    def restore_state_from_file(self, weight_file):
        """
        Restore model using weights stored from a previous run.
        
        Parameters : weight_file
        
        Outputs : 
            
        Returns : None
        """
        # Open a file in read-binary mode
        with open(weight_file, 'rb') as f:
            print('Restoring state from', weight_file)

            # torch interprets the file, then we can access using string keys
            checkpoint = torch.load(f)
            
            # load network weights
            self.model_accs.load_state_dict(checkpoint['state_dict'])
            
            # if optim is provided, load the state of the optim
            if hasattr(self, 'optimizer'):
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            # load iteration count
            self.iteration = checkpoint['global_step']
        
        print('Restoration complete.')
    
    def save_state(self,best=False):
        """
        Save model weights to a file.
        
        Parameters : best
        
        Outputs : 
            
        Returns : filename
        """
        filename = "{}{}{}{}".format(self.dirpath,
                                     str(self.model._get_name()),
                                     ("BEST3dnew1" if best else ""),
                                     ".pth")
        
        # Save model state dict in appropriate from depending on number of gpus
        model_dict = self.model_accs.state_dict()
        
        # Save parameters
        # 0+1) iteration counter + optimizer state => in case we want to "continue training" later
        # 2) network weight
        torch.save({
            'global_step': self.iteration,
            'optimizer': self.optimizer.state_dict(),
            'state_dict': model_dict
        }, filename)
        print('Saved checkpoint as:', filename)
        return filename
