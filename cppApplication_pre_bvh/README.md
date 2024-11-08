# c++ implementation, old version before bvh but with new shading model.

## Steps to build and run

### Install docker

<https://docs.docker.com/desktop/>

### Run commands in the cppapplication directory

```bash
sudo docker build -t cppapplication .
sudo docker run -i -v {model dir path}:/models cppapplication /cppapplication/build/ascii_rt /models/pirateship.obj
```
