from vae import LSTMVAE, VAEAugmenter
import numpy as np

data = np.random.normal(size=(40, 1, 100))

vae = LSTMVAE(series_len=100)
augmenter = VAEAugmenter(vae)
augmenter.fit(data, epochs=10, batch_size=64)

samples = augmenter.sample(n=20)
print(samples.shape)