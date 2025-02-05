# HPC usage

In this course, you will be assigned with assignments requiring neural network training with GPU.  We use the NYU HPC in this course for the GPU computing resources.  This instruction will help you with the usage of NYU HPC in this course.

Everyone in the course is assigned with 200 GPU hours and sufficient CPU time. The following partitions are allowed to use:

```
CSCI_GA_2590_2025sp = {
  accounts = { "csci_ga_2590-2025sp" },
  partitions = { "interactive", "n1s8-v100-1", "n1s16-v100-2", "n2c48m24",
                 "g2-standard-12", "g2-standard-24", 
                 "c12m85-a100-1", "c24m170-a100-2" }
}
```

All the following operations should be done within the NYU network environment, because the servers have internal IP address and cannot be accessed from the public Internet. If you are not at campus, you can use the VPN.



## NYU HPC Greene

Each of you should already have access to the NYU HPC. To use the HPC, users login to the Greene cluster first. Instructions are available from https://sites.google.com/nyu.edu/nyu-hpc/accessing-hpc?authuser=0.

Below are some important steps selected from the instruction:

1. (optional, if you are under NYU network, either a campus wifi or NYU VPN, you can skip this)  In your command line, use the the following ssh command.

   ```bash
   ssh <NetID>@gw.hpc.nyu.edu # When prompted, enter the password associated with your NYU NetID.
   ```

2. Use this ssh command to access the greene log-in machine:

   ```bash
   # this will connect you to Greene HPC cluster
   ssh <NetID>@greene.hpc.nyu.edu
   ```



## Burst

From one Greene login node, run `ssh burst`, and you will be connected to the log-burst node. You can also directly run `ssh <NetID>@log-burst.hpc.nyu.edu` from you terminal to access it without accessing greene log-in node first.

On this log-burst node, you can launch instances with GPU.  For example, if you want to launch a simple CPU only interactive job for 4 hours, you can type this following command in the log-burst node:

```bash
srun --account=csci_ga_2590-2025sp --partition=interactive --time=04:00:00 --pty /bin/bash
```

This command will open up a shell in the target partition/instance for you, and you can use that shell in your current terminal window.

Other examples:

```bash
# A GPU job with 1 V100 GPU for 4 hours
srun --account=csci_ga_2590-2025sp --partition=n1s8-v100-1 --gres=gpu:v100:1 --time=04:00:00 --pty /bin/bash

# A GPU job with 2 V100 GPUs for 4 hours 
srun --account=csci_ga_2590-2025sp --partition=n1s16-v100-2 --gres=gpu:2 --pty /bin/bash
```

Greene Data transfer nodes is available with hostname greene-dtn. On a Cloud instances, run `scp`, for example:

```bash
scp -rp greene-dtn:/scratch/work/public/singularity/ubuntu-20.04.3.sif .
```



## Conda environment

Greene has a limited inode resource for each of your `$HOME` directory, so you may not put a huge amount of files like the conda environment in your home directory. 

The way to fix this and use conda is through singularity and overlay file.

Instructions to setup Conda enviorment with Singularity and overlay file: https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene/software/singularity-with-miniconda. Overlay file templates are available from `/share/apps/overlay-fs-ext3`. Singularity OS images are available from `/share/apps/images`. 

First you may copy over the empty filesystem image to put the conda environment later (you only need to do this once semester)

```bash
# On Burst: first get on GCP
srun --account=csci_ga_2590-2025sp --partition=n1s8-v100-1 --gres=gpu --time=1:00:00 --pty /bin/bash
# Then download the overlay filesystem
cd /scratch/[netid]
scp greene-dtn:/scratch/work/public/overlay-fs-ext3/overlay-25GB-500K.ext3.gz .

# Unzip the ext3 filesystem. May take 5 min here.
gunzip -vvv ./overlay-25GB-500K.ext3.gz
```

Filesystems can be mounted as read-write (`rw`) or read-only (`ro`) when we use it with singularity.
- read-write: use this one when setting up env (installing conda, libs, other static files)
- read-only: use this one when running your jobs. It has to be read-only since multiple processes will access the same image. It will crash if any job has already mounted it as read-write.

Now let's launch singularity container with the fresh filesystem we just copied over (you need to do the below every time you want to run GPU jobs):

```bash
# On GCP (assuming our current directory is /scratch/[netid])
# Copy the appropriate singularity image to the current working directory
scp -rp greene-dtn:/scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif .

singularity exec --bind /scratch --nv --overlay /scratch/[netid]/overlay-25GB-500K.ext3:rw /scratch/[netid]/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif /bin/bash
```

**Important**: if you want to use GPUs inside singularity, add --nv argument after exec.

We are going to install Conda package in the `/ext3/` folder where your own filesystem is mounted.

```
## On GCP
Singularity> cd /ext3/
Singularity> wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
--2023-09-24 14:31:12--  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
Resolving repo.anaconda.com (repo.anaconda.com)... 104.16.130.3, 104.16.131.3, 2606:4700::6810:8203, ...
Connecting to repo.anaconda.com (repo.anaconda.com)|104.16.130.3|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 103219356 (98M) [application/x-sh]
Saving to: 'Miniconda3-latest-Linux-x86_64.sh'

Miniconda3-latest-Linux-x86_64.sh             100%[==============================================================================================>]  98.44M   186MB/s    in 0.5s

2023-09-24 14:31:13 (186 MB/s) - 'Miniconda3-latest-Linux-x86_64.sh' saved [103219356/103219356]
```

Then you can install conda and utilize the environment and start coding.



## Optional Instructions for `ssh` Pro Users

This part is not required for you to be able to use the HPC. But if you are familiar with ssh and you want to for example connect VSCode to the instances, you can take a look at this part. But keep in mind, you need to know what you are doing rather than copy the commands. Otherwise, you can stick to the part above. 

First you need to have your public and private key pair in order to use ssh without password. This can be done by `ssh-kengen` command if you do not have one. After the keygen, you will find the private key at `~/.ssh/id_rsa` and the public key at `~/.ssh/id_rsa.pub`. Keep the private key to yourself, and put your public key at the `~/.ssh/authorized_keys` file in the servers (both greene and burst)

Next, you can set up the `~/.ssh/config` file in your laptop. For example, you can have

```
Host greene
    HostName greene.hpc.nyu.edu
    User NetID
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null

Host burst
    HostName log-burst.hpc.nyu.edu
    User NetID
```

If you want to directly connect to the instance, either via ssh or VSCode, you need to use `sbatch` rather than `srun` to open up an instance. For example, launching a GPU node for one hour can be

```bash
sbatch --account=csci_ga_2590-2025sp --partition=n1s8-v100-1 --gres=gpu:v100:1 --time=01:00:00 --wrap "sleep infinity"
```

Then you can use `squeue --me` to see if your instance is launched.

For example,

```bash
[NetID@log-burst ~]$ squeue --me
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
            167330 n1s8-v100     wrap   hj2533  R       0:41      1 b-3-17
```

When the `ST` field is R, it means the instance is ready, and you can connect to it. In order to connect to the instance from the local computer, you need to add the hostname to your ssh config file, with the proxy field using burst login node. For the above example, it can be

```
Host foobar
    HostName b-3-17
    User NetID
    ProxyJump burst
```

And now use `ssh foobar` from you *local* shell to access the computing instance.

\[Note:\] letting VSCode connect directly into the singularity container running on the burst is possible, but requires advanced manipulation of ssh config.  It is NOT recommended so I will not provide the method here.  You can explore how to accomplish it by yourself (you are expected to be able to set it up without external help if you really need this feature.)
