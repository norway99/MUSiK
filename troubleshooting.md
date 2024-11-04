## Troubleshooting
### Installation
**Be sure to use Python 3.10 to avoid dependency conflicts.**
#### For submodule k-wave-python
The k-wave-python package is a submodule here, so when you clone this repository, you can use
```
git clone --recursive --remote-submodules git@github.com:norway99/MUSiK.git
```
or if you have already cloned the MUSik repo, use this
```
git submodule update --init --recursive --remote
```
Remember to add the **remote** flag, because the k-wave package is updating frequently, and there are deprecated dependencies that may cause installation failure.
#### Running for the first time
k-wave needs to download some binaries at the first run. It may cause issue when you are using cloud computating resources (like SuperCloud) whose computation nodes do not have internet access.
I am using SuperCloud and there is a ["download" partition](https://mit-supercloud.github.io/supercloud-docs/using-the-download-partition/) with data transfer nodes.
So my strategy was to use this "download" partition and jupyter notebook in turn to download the binaries and target uninstalled dependencies.
