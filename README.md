# MUSiK: Multi-transducer Ultrasound Simulations in K-wave, DEVELOPMENT EDITION


## Developer notes

Clone the public project, make a dev branch, and add the private remote as follows:

```shell
$ git clone git@github.com:norway99/MUSiK.git
$ git push -u origin main
$ git branch -b dev 
$ git remote add private git@github.com:norway99/musik_dev_private.git
$ git push -u private dev
```
Now whenever you are on branch dev you will automatically push to and pull from the private remote.

Switch branches:

```shell
$ git checkout <branch> [OR]
$ git switch <branch>
```

Find the list of branches and see which one you're on:

```shell
$ git branch```

Add/update single files from main while on branch dev:

```shell
$ git checkout main <filename>```

Can also be run in reverse (grab single files from dev while on branch main). Bear in mind that these changes still need to be committed/pushed.

Upload all changes from branch dev to main:

```shell
$ git checkout main
$ git merge dev
```

and resolve any merge conflicts that might arise.


## Installation

Clone the repository:
```shell
$ git clone https://github.com/norway99/MUSiK.git 
$ cd MUSiK
```

Install packages into a python environment:
```shell
$ pip install -r requirements.txt
```

Install k-wave-python: 
```shell
$ git clone https://github.com/waltsims/k-wave-python.git ./
```
