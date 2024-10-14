# cuda_project c++ implementation

## Steps to build and run

### Install docker

<https://docs.docker.com/desktop/>

### Run commands in the cppapplication directory

```bash
sudo docker build -t cppapplication .
sudo docker run -i -v /home/ssyverson/Documents/Code/cuda_project/models:/models cppapplication /cppapplication/build/ascii_rt /models/teapot.obj
```
