import keras
from keras import backend as K
from keras.layers import (Dense, Input, Flatten)
from keras.layers import Lambda, Conv2D
from keras.models import Model
from keras.layers import Reshape, Conv2DTranspose
from keras.losses import mse
###########################################
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

inner_dim = 16
latent_dim = 6

image_size = (64,78,1)###########################################
###########################################
inputs = Input(shape=image_size, name='encoder_input')
x = inputs

x = Conv2D(32, 3, strides=2, activation='relu', padding='same')(x)
x = Conv2D(64, 3, strides=2, activation='relu', padding='same')(x)

# shape info needed to build decoder model
shape = K.int_shape(x)

# generate latent vector Q(z|X)
x = Flatten()(x)
x = Dense(inner_dim, activation='relu')(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(inner_dim, activation='relu')(latent_inputs)
x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(x)
x = Reshape((shape[1], shape[2], shape[3]))(x)

x = Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same')(x)
x = Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same')(x)

outputs = Conv2DTranspose(filters=1, kernel_size=3, activation='sigmoid', padding='same', name='decoder_output')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae')

def vae_loss(x, x_decoded_mean):
    reconstruction_loss = mse(K.flatten(x), K.flatten(x_decoded_mean))
    reconstruction_loss *= image_size[0] * image_size[1]
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    return vae_loss

optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000)###########################################
vae.compile(loss=vae_loss, optimizer=optimizer)
vae.fit(train_X, train_X,
        epochs=500,
        batch_size=128,
        verbose=1,
        shuffle=True,
        validation_data=(valid_X, valid_X))