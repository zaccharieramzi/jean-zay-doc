# Tips and Tricks

## Python

### Install miniconda (recommended solution if you are already familiar with conda)

Install `miniconda` in `$WORK/miniconda3`:
```sh
# download Miniconda installer
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
# install Miniconda
MINICONDA_PATH=$WORK/miniconda3
chmod +x miniconda.sh && ./miniconda.sh -b -p $MINICONDA_PATH
# make sure conda is up-to-date
source $MINICONDA_PATH/etc/profile.d/conda.sh
conda update --yes conda
# Update your .bashrc to initialise your conda base environment on each login
conda init
```

If you run out of space or inodes on `$WORK` (`irdquota -w` can help you
figuring out whether you are close to the limit) you can send an email to
[assist@idris.fr](mailto:assist@idris.fr) and ask for an increase. Try
something between 5x-10x with some small justification and that should go
through without too much problem (if that's not the case, open an
[issue](https://github.com/jean-zay-users/jean-zay-doc/issues/new) to improve
this doc!).

## SLURM

### How to launch an interactive job

Your can use `srun` to launch an interactive job.

For example, if you want to use a node with 4 GPUs during 1
hour, you can type:
```
srun --ntasks=1 --gres=gpu:4 --time=01:00:00 --pty bash -i
```

Now, you have a brand new shell on a compute node where you can run your scripts interactively
during 1h.


## Miscellaneous

### Managing your data and the storage spaces

Be careful about the place where you put your data on the JZ super-computer,
since there are quotas for each project, depending on the storage space and the
number of files (inodes) that you use. Additionally, some spaces are temporary.

There is a detailed description of the storage spaces [here](http://www.idris.fr/jean-zay/cpu/jean-zay-cpu-calculateurs-disques.html).

Briefly:

- `$HOME` (3Gb) -> for config files.
- `$WORK` (limited on inodes) -> for code, (small) databases.
- `$SCRATCH` (very large limits, temporary) -> output data, large databases
- `$STORE` (large space, occasional consultation)  -> permanent large databases.
- `$DSDIR` (popular databases on demand).

It is worth noting that `$SCRATCH` consists of a farm of SSD storage
devices and it provides the best performance in reading/writing operations.
You must also be aware that `$SCRATCH` is regularly "cleaned": files that
have not been accessed (i.e. at least read) for 30 days are definitely
removed. So you **risk to lose your data** if your keep it there without using it.

You can consult your disk quota anytime with the command `idrquota` (see
`idrquota -h`). If you need more space or inodes on your personal spaces
(`$WORK` or `$STORE`), just ask the support team at
[assist@idris.fr](mailto:assist@idris.fr).

If you need to send data to Jean-Zay a good idea is to use `rsync`. E.g.:

```
rsync -avz /your/local/database/ your-jean-zay-login@jean-zay:/gpfsscratch/your/remote/dir/
```

### Connect seamlessly from your local machine

Add local your public ssh key to the `~/.ssh/authorized_keys` of your account on the jean-zay cluster.
Your local public ssh key can be found in `~/.ssh/id_rsa.pub`.

In your local ssh configuration, found in `~/.ssh/config`, you can also add the following:
```
Host jz
hostname jean-zay.idris.fr
user <user-name>
```

To connect to the jean-zay cluster you will then just need to do `ssh jz`.
