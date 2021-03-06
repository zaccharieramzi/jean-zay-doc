# [MNIST with TensorFlow using MPI through Horovod](https://github.com/jean-zay-users/jean-zay-doc/tree/master/docs/examples/tf/tf_mpi)

This toy-example shows how to do distributed training using TensorFlow and
Horovod.  Since Jean-Zay nodes are connected using MPI and with NCCL,
synchronous training is theoretically well scalable.

# Submit the job

## A few things

Look at the slurm script, `tf_mpi_mnist.job`. 
There is a few important things:

 - `#SBATCH --partition=gpu_p1`: We use the `gpu_p1` partition, which has nodes
   with 4 GPUs each,
 - `#SBATCH --ntasks-per-node=4`: We set 4 MPI tasks per node: one task per
   GPU,
 - `#SBATCH --cpus-per-task=10`: Since nodes have 40 CPUs each, we have to ask
   slurm to allocate 1/4 of the node resources (i.e. 10)
 - `#SBATCH --ntasks=32`: We ask for a total of 32 MPI tasks. Since we have 4
   tasks per node, this means that we will use 8 nodes.

## Enjoy

```
cd jean-zay-doc/docs/examples/tf/tf_mpi
sbatch tf_mpi_mnist.job
```

## Code

The code for the distributed training (`tf_mpi_mnist.py`):

{{code_from_file("examples/tf/tf_mpi/tf_mpi_mnist.py", "python")}}

and the script to launch the job:

{{code_from_file("examples/tf/tf_mpi/tf_mpi_mnist.job", "bash")}}

