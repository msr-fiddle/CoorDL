# CoorDL
 
Installing or building CoorDL
---------------------------
- `git clone https://github.com/jayashreemohan29/CoorDL.git`
- `git submodule sync --recursive && \git submodule update --init --recursive`
- `cd $DALI_ROOT/docker`
- `CREATE_RUNNER="YES" ./build.sh`
- `nvidia-docker run --ipc=host --mount src=/,target=/datadrive/,type=bind -it --rm --network=host --cpus=24 -m 200g --privileged nvidia/dali:py36_cu10.run`

