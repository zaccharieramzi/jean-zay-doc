# all taken from https://www.tensorflow.org/guide/keras/functional
import click


@click.command()
def train_dense_model_click():
    return train_dense_model(batch_size=16)


def train_dense_model(batch_size):
    # limit imports oustide the call to the function, in order to launch quickly
    # when using dask
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    from fastmri_recon.models.utils.fourier import IFFT
    from fastmri_recon.models.utils.fastmri_format import general_fastmri_format
    # model building
    tf.keras.backend.clear_session()  # For easy reset of notebook state.

    class MyModel(keras.models.Model):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.conv = layers.Conv2D(1, 3, padding='same')
            self.ifft = IFFT(False)

        def call(self, inputs):
            kspace, mask = inputs
            image = self.ifft(kspace)
            image = general_fastmri_format(image)
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
    x_train = (
        tf.cast(tf.random.normal([16*10, 320, 320, 1]), tf.complex64),
        tf.cast(tf.random.normal([16*10, 320, 320, 1]), tf.complex64),
    )
    y_train = tf.random.normal([16*10, 320, 320, 1])
    ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(16).repeat()

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=2,)
    return True

if __name__ == '__main__':
    train_dense_model_click()
