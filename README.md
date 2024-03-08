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
Now whenever you are on branch dev (```shell $ git checkout dev```) you will automatically push to and pull from the private remote.

Find out what branch you're on by running ```shell $ git branch```.

To add/update single files from master while on branch dev, run ```shell $ git checkout master <filename>```. Can also be run in reverse (grab single files from dev while on branch master). Bear in mind that these changes still need to be committed/pushed.

To upload all changes from branch dev to master:

```shell
$ git checkout master
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
