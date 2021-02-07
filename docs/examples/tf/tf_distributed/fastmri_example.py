# all taken from https://www.tensorflow.org/guide/keras/functional
import click


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

    from fastmri_recon.config import FASTMRI_DATA_DIR
    from fastmri_recon.data.datasets.multicoil.fastmri_pyfunc import train_masked_kspace_dataset_from_indexable
    from fastmri_recon.models.utils.fourier import IFFT
    from fastmri_recon.models.utils.fastmri_format import general_fastmri_format
    # model building
    tf.keras.backend.clear_session()  # For easy reset of notebook state.

    class MyModel(keras.models.Model):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.conv = layers.Conv2D(1, 3, padding='same')
            self.ifft = IFFT(False, multicoil=True)

        def call(self, inputs):
            kspace, mask, smaps = inputs
            image = self.ifft([kspace, smaps])
            image = general_fastmri_format(image)
            # to check that splitting happens correctly
            tf.print(tf.shape(image))
            image = self.conv(image)
            return image

    slurm_resolver = tf.distribute.cluster_resolver.SlurmClusterResolver(port_base=15000)
    mirrored_strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=slurm_resolver)
    print('Number of replicas:', mirrored_strategy.num_replicas_in_sync)
    with mirrored_strategy.scope():
        model = MyModel(name='fastmri_model')

        model.compile(loss='mse', optimizer=keras.optimizers.RMSprop())

    # training and inference
    # x_train = (
    #     tf.cast(tf.random.normal([16*50, 320, 320, 1]), tf.complex64),
    #     tf.cast(tf.random.normal([16*50, 320, 320, 1]), tf.complex64),
    # )
    # y_train = tf.random.normal([16*50, 320, 320, 1])
    # ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(16).repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    path = Path(FASTMRI_DATA_DIR) / 'multicoil_train'
    def _dataset_fn(input_context):
            ds = train_masked_kspace_dataset_from_indexable(
                str(path) + '/',
                input_context=input_context,
                inner_slices=None,
                rand=True,
                scale_factor=1e6,
                batch_size=16 // input_context.num_input_pipelines,
                target_image_size=(640, 400),
                parallel=False,
            )
            options = tf.data.Options()
            options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
            ds = ds.with_options(options)
            return ds
    ds = mirrored_strategy.distribute_datasets_from_function(_dataset_fn)

    history = model.fit(ds, steps_per_epoch=50, epochs=2,)
    return True

if __name__ == '__main__':
    train_dense_model_click()
