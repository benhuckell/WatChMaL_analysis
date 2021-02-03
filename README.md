# WatChMaL_analysis
Example repository for usage of the WatChMaL repository

## Working With Submodules

See for reference: https://git-scm.com/book/en/v2/Git-Tools-Submodules

### Key commands

#### Cloning a repo with submodules

To clone the repo with submodules included:

```
$ git clone --recurse-submodules https://github.com/Whisky-Jack/WatChMaL_analysis.git
```

FOR NOW JUST TREAT THE SUBMODULE AS ITS OWN REPO, AND CD TO THE SUBMODULE DIRECTORY TO PUSH/PULL CHANGES

#### Fetching updates from remote

To fetch remote update the submodules you can:

1. Enter the submodule directory and run fetch then merge. e.g.

```
$ cd WatChMaL
$ git fetch
$ git merge origin/master
```

2. You can get Git to fetch updates for the submodule for you using

```
$ git submodule update --remote WatChMaL
```


#### Pushing local changes to a submodule

To push local changes to a submodule you first need to add to stage the changes to the submodule in the main repo. You can then:

1. Push local changes to the repo and check whether there are unpushed submodule changes using:

```
$ git push --recurse-submodules=check
```

This will warn you if any local submodule changes haven't been pushed, and give you instructions for how to push these changes. This should be done at a minimum to make sure you aren't pushing a version of the repo relying on unpushed local changes to the submodule.

2. Push changes to the repo and try to automatically push changes to the submodule using:

```
$ git push --recurse-submodules=on-demand
```

3. As before you can enter the submodule directory and use normal git commands:

```
$ cd WatChMaL
$ git push
```