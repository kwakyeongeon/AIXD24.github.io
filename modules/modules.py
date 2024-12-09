import tensorflow as tf
from tensorflow.keras import layers, models

def build_model():

    # Encoder 모델
    latent_dim = 2  # 잠재 공간의 차원
    encoder_input = layers.Input(shape=(2,))
    encoded = layers.Dense(64, activation='relu')(encoder_input)
    latent = layers.Dense(latent_dim, activation='linear')(encoded)

    encoder = models.Model(encoder_input, latent, name="encoder")

    # Decoder 모델
    decoder_input = layers.Input(shape=(latent_dim,))
    decoded = layers.Dense(64, activation='relu')(decoder_input)
    output = layers.Dense(2, activation='linear')(decoded)

    decoder = models.Model(decoder_input, output, name="decoder")

    # Autoencoder 모델
    autoencoder_input = layers.Input(shape=(2,))
    encoded_latent = encoder(autoencoder_input)
    decoded_output = decoder(encoded_latent)

    autoencoder = models.Model(autoencoder_input, decoded_output, name="autoencoder")

    # 모델 컴파일
    autoencoder.compile(optimizer='adam', loss='mse', metrics=["mae"])

    return autoencoder