# [TensorFlow single node examples](https://github.com/jean-zay-users/jean-zay-doc/tree/master/docs/examples/tf/tf_simple)

To run the examples you will need to first install `click` in your environment.
```
module load python/3.7.5 &&\
pip install click
```

Then you need to clone the jean-zay repo in your `$WORK` dir:
```
cd $WORK &&\
git clone https://github.com/jean-zay-users/jean-zay-doc.git
```

## Classical examples

For the single GPU job you can do:
```
cd jean-zay-doc/docs/examples/tf/tf_simple
sbatch mnist_submission_script.slurm
```

For the multi GPU job you can do:
```
jean-zay-doc/docs/examples/tf/tf_simple
sbatch mnist_submission_script_multi_gpus.slurm
```

The training code used in this example is:

{{code_from_file("examples/tf/tf_simple/mnist_example.py", "python")}}

and the script used to launch a single GPU job is:

{{code_from_file("examples/tf/tf_simple/mnist_submission_script.slurm", "bash")}}

to launch the same code using a multiGPU configuration, use the following script:

{{code_from_file("examples/tf/tf_simple/mnist_submission_script_multi_gpus.slurm", "bash")}}

## Dask example

To run the dask example you will need to install `dask-jobqueue` in your
environment additionally.  Notice that this time you need to use the python
module with tensorflow loaded, because [dask will by default use the same
python for the worker as the one you used for the
scheduler](https://jobqueue.dask.org/en/latest/debug.html).  See this [GitHub
issue](https://github.com/dask/dask-jobqueue/issues/408) for more information.
```
module load tensorflow-gpu/py3/2.1.0 &&\
pip install click dask-jobqueue
```

You can then do:
```
python jean-zay-doc/docs/examples/tf/tf_simple/dask_script.py 64
```

where 64 is the batch size you want to run the mnist example with.  If you want
multiple batch sizes just have them space-separated.

Be sure to load the tensorflow module before launching the dask script because
otherwise Tensorflow will not be loaded.  This is because the python executable
used to launch the dask worker is the same as the one used to launch the
scheduler by default.  You can set it otherwise in the cluster if you want
something more tailored.

Here is the code for the file `dask_script.py`:

{{code_from_file("examples/tf/tf_simple/dask_script.py", "python")}}
