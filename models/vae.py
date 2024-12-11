import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from torch import Tensor


class VAE:

    def __init__(self):
        self.model: AutoencoderKL = torch.load('cache/vae.pt', weights_only=False)
        self.scaling_factor = self.model.config.scaling_factor
        self.shift_factor = self.model.config.shift_factor
        self.latent_channels = self.model.config.latent_channels

    def empty_cache(self, device: str):
        if device == 'cuda':
            torch.cuda.empty_cache()
        elif device == 'mps':
            torch.mps.empty_cache()

    def process_in(self, z: Tensor) -> Tensor:
        if self.shift_factor is not None:
            z = z - self.shift_factor
        z = z * self.scaling_factor
        return z

    def process_out(self, z: Tensor) -> Tensor:
        z = z / self.scaling_factor
        if self.shift_factor is not None:
            z = z + self.shift_factor
        return z

    @torch.inference_mode()
    def encode(self, x: Tensor) -> Tensor:
        self.model.encoder = self.model.encoder.to(x.device)
        z = []
        for xi in x:
            posterior: DiagonalGaussianDistribution = self.model.encode(xi[None]).latent_dist
            z.append(posterior.sample())
        z = torch.cat(z)
        z = self.process_in(z)
        self.model.encoder = self.model.encoder.to('cpu')
        # self.empty_cache(x.device.type)
        return z

    @torch.inference_mode()
    def encode_batch(self, x: Tensor) -> Tensor:
        self.model.encoder = self.model.encoder.to(x.device)
        posterior: DiagonalGaussianDistribution = self.model.encode(x).latent_dist
        z = posterior.sample()
        z = self.process_in(z)
        self.model.encoder = self.model.encoder.to('cpu')
        # self.empty_cache(x.device.type)
        return z

    @torch.inference_mode()
    def decode(self, z: Tensor) -> Tensor:
        self.model.decoder = self.model.decoder.to(z.device)
        z = self.process_out(z)
        y = torch.cat([self.model.decode(zi[None]).sample for zi in z])
        self.model.decoder = self.model.decoder.to('cpu')
        # self.empty_cache(z.device.type)
        return y

    @torch.inference_mode()
    def __call__(self, x: Tensor) -> Tensor:
        self.model = self.model.to(x.device)
        y = []
        for xi in x:
            posterior: DiagonalGaussianDistribution = self.model.encode(xi[None]).latent_dist
            zi = posterior.sample()
            y.append(self.model.decode(zi).sample)
        y = torch.cat(y)
        self.model = self.model.to('cpu')
        # self.empty_cache(x.device.type)
        return y
