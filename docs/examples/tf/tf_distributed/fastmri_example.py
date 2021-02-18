# all taken from https://www.tensorflow.org/guide/keras/functional
import click
from tensorflow.keras.callbacks import ModelCheckpoint

# taken from https://github.com/tensorflow/tensorflow/issues/39679
class ModelCheckpointWorkAround(ModelCheckpoint):
    def set_model(self, model):
        # Work around, so that the if at
        # https://github.com/tensorflow/tensorflow/blob/1186e3f2098793952aa82bf356dfe51b967fb26c/tensorflow/python/keras/callbacks.py#L1189
        # is skipped, so that self.save_weights_only remains False.
        self.model = model


@click.command()
def train_dense_model_click():
    return train_dense_model(batch_size=16)


def train_dense_model(batch_size):
    # limit imports oustide the call to the function, in order to launch quickly
    # when using dask
    from pathlib import Path

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.mixed_precision import experimental as mixed_precision

    from fastmri_recon.config import FASTMRI_DATA_DIR, CHECKPOINTS_DIR, TMP_DIR
    from fastmri_recon.data.datasets.multicoil.fastmri_pyfunc import train_masked_kspace_dataset_from_indexable
    from fastmri_recon.models.subclassed_models.denoisers.proposed_params import get_model_specs
    from fastmri_recon.models.subclassed_models.xpdnet import XPDNet
    from fastmri_recon.models.utils.fourier import IFFT
    from fastmri_recon.models.utils.fastmri_format import general_fastmri_format
    # model building
    tf.keras.backend.clear_session()  # For easy reset of notebook state.

    slurm_resolver = tf.distribute.cluster_resolver.SlurmClusterResolver(port_base=15000)
    mirrored_strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=slurm_resolver)
    print('Number of replicas:', mirrored_strategy.num_replicas_in_sync)

    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)

    run_params = {
        'n_primal': 5,
        'multicoil': False,
        'n_scales': 4,
        'n_iter': 1,
        'refine_smaps': False,
        'res': True,
        'output_shape_spec': False,
        'multi_gpu': False,
        'refine_big': False,
        'primal_only': True,
        'n_dual': 1,
        'n_dual_filters': 8,
        'multiscale_kspace_learning': False,
    }
    n_primal = 5
    model_fun, model_kwargs, n_scales, res = [
         (model_fun, kwargs, n_scales, res)
         for m_name, m_size, model_fun, kwargs, _, n_scales, res in get_model_specs(n_primal=n_primal, force_res=False)
         if m_name == 'MWCNN' and m_size == 'small'
    ][0]

    with mirrored_strategy.scope():
        model = XPDNet(model_fun, model_kwargs, **run_params)

        model.compile(loss='mse', optimizer=keras.optimizers.RMSprop())

    # training and inference
    x_train = (
        tf.cast(tf.random.normal([16*50, 320, 320, 1]), tf.complex64),
        tf.cast(tf.random.normal([16*50, 320, 320, 1]), tf.complex64),
    )
    y_train = tf.random.normal([16*50, 320, 320, 1])
    ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(16).repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # path = Path(FASTMRI_DATA_DIR) / 'multicoil_train'
    # def _dataset_fn(input_context):
    #         ds = train_masked_kspace_dataset_from_indexable(
    #             str(path) + '/',
    #             input_context=input_context,
    #             inner_slices=None,
    #             rand=True,
    #             scale_factor=1e6,
    #             batch_size=16 // input_context.num_replicas_in_sync,
    #             target_image_size=(640, 400),
    #             parallel=False,
    #         )
    #         options = tf.data.Options()
    #         options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    #         ds = ds.with_options(options)
    #         return ds
    # ds = mirrored_strategy.distribute_datasets_from_function(_dataset_fn)
    if slurm_resolver.task_id == 0:
        chkpt_path = f'{CHECKPOINTS_DIR}test_checkpoints/test' + '-{epoch:02d}.h5'
    else:
        chkpt_path = f'{TMP_DIR}test_checkpoints/test' + '-{epoch:02d}.h5'
    chkpt_cback = ModelCheckpointWorkAround(
        chkpt_path,
        save_freq=100,
        save_weights_only=True,
    )
    history = model.fit(ds, steps_per_epoch=50, epochs=2, callbacks=[chkpt_cback])
    return True

if __name__ == '__main__':
    train_dense_model_click()
