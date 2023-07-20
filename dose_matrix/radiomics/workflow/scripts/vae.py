
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class UnFlatten(nn.Module):
    # NOTE: (size, x, x, x) are being computed manually as of now (this is based on output of encoder)
    def forward(self, input, size=512): # size=128
        return input.view(input.size(0), size, 3, 3, 3)
        # return input.view(input.size(0), size, 6, 6, 6)


class CVAE_3D_N128(nn.Module):
    """
    3D Convolutional Variational Autoencoder.

    Parameters
    ==========

    image_channels: number of channels of the image (default=1, gray level dose distribution image)
    input_image_size: the expected shape of each image input
    z_dim: dimension of the latent space (the multinormal dimension)
    """
    def __init__(self, image_channels = 1, z_dim = 32, input_image_size = None):
        super(CVAE_3D_N128, self).__init__()

        # Input size
        assert input_image_size is not None
        self.input_image_size = np.asarray(input_image_size).astype(int)
        assert self.input_image_size.shape == (3,)
        # Encoder part
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=image_channels, out_channels=16, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=16),
            nn.ReLU(),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=32),
            nn.ReLU(),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=64),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=128),
            nn.ReLU(),
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=128),
            nn.ReLU(),
            nn.Flatten() # reshape layer
        )

        # Encode to the latent space of normal distributions
        # z_dim is set by user
        # h_dim is the dimension of the encoder's last layer
        # mu_x = encoder(fc1(x)) and sigma_x = encoder(fc2(x))
        n_channels_end = 128
        n_conv3d_layers = 5
        output_image_shape = [int(v) for v in self.input_image_size - n_conv3d_layers*3]
        h_dim = n_channels_end * np.prod(output_image_shape) # - nb_conv3D*(kernel_size-1)
        self.h_dim = h_dim
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        # fc3(z ~ N(mu_x, sigma_x)) => decoder
        self.fc3 = nn.Linear(z_dim, h_dim)

        # Decoder part
        unflatten_size = [n_channels_end] + output_image_shape
        self.unflatten_size = unflatten_size
        self.decoder = nn.Sequential(
            nn.Unflatten(1, unflatten_size),
            nn.BatchNorm3d(num_features=128),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=128, out_channels=128, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=128),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=64),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=32),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=16),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=16, out_channels=image_channels, kernel_size=4, stride=1, padding=0), # dimensions should be as original
            nn.BatchNorm3d(num_features=image_channels),
            # nn.Sigmoid(),

            # if it does not work without sigmoid:
                # check another batchnorm or relu
                # recover original dims: use nn.linear and reshape to original size
                # nn.conv3d with kernel size equal to input size
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        # z = mu + std * eps
        z = eps.mul(std).add_(mu)
        return z

    def bottleneck(self, h):
        # print("[INFO] bottleneck h size:", h.size())
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        # print("[INFO] Input data shape:", x.size())
        h = self.encoder(x)
        # print("[INFO] h size:", h.size(), " estimated h size:", self.h_dim)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def encode_debug(self, x):
        print("[INFO] Input size:", self.input_image_size)
        print("[INFO] Begin: Input data shape:", x.size())
        x = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=4, stride=1, padding=0)(x)
        x = nn.BatchNorm3d(num_features=16)(x)
        x = nn.ReLU()(x)
        print("[INFO] Input data shape:", x.size())
        x = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=4, stride=1, padding=0)(x)
        x = nn.BatchNorm3d(num_features=32)(x)
        x = nn.ReLU()(x)
        print("[INFO] Input data shape:", x.size())
        x = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=4, stride=1, padding=0)(x)
        x = nn.BatchNorm3d(num_features=64)(x)
        x = nn.ReLU()(x)
        print("[INFO] Input data shape:", x.size())
        x = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=4, stride=1, padding=0)(x)
        x = nn.BatchNorm3d(num_features=128)(x)
        x = nn.ReLU()(x)
        print("[INFO] Input data shape:", x.size())
        x = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=4, stride=1, padding=0)(x)
        x = nn.BatchNorm3d(num_features=128)(x)
        x = nn.ReLU()(x)
        print("[INFO] Input data shape:", x.size())
        x = nn.Flatten()(x) # reshape layer
        print("[INFO] Input data shape:", x.size())

        return x

    def decode(self, z):
        z = self.decoder(z)
        return z

    def decode_debug(self, z):
        print(f"z shape: {z.shape}")
        z = nn.Unflatten(1, self.unflatten_size)(z)
        print(f"z shape unflatttened: {z.shape}")
        z = nn.BatchNorm3d(num_features=128)(z)
        print(f"z shape batch: {z.shape}")
        z = nn.ReLU()(z)
        z = nn.ConvTranspose3d(in_channels=128, out_channels=128, kernel_size=4, stride=1, padding=0)(z)
        z = nn.BatchNorm3d(num_features=128)(z)
        z = nn.ReLU()(z)
        print(f"z shape batch: {z.shape}")
        return z

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        # Sending data to appropriate device
        device = next(self.parameters()).device
        x = x.to(device)

        # Step 1: compute representation (fetch it separately for later clustering)
        z_representation = self.representation(x)
        # print("[INFO] Forward z_representation:", z_representation.size())
        # print("[INFO] Reshaped latent z", z_representation.view(z_representation.size(0), 8, 8).size())

        # Step 2: call full CVAE --> encode & decode
        z, mu, logvar = self.encode(x)
        z = self.fc3(z)
        # print("[INFO] Latent z after dense fc:", z.size())
        # print("[INFO] mu:", mu.size())
        # print("[INFO] logvar", logvar.size())
        x_hat = self.decode(z)
        # print(f"[INFO] Input shape: {self.input_image_size}, input x: {x.shape}, reconstructed x: {x_hat.shape}")

        return x_hat, mu, logvar, z_representation

class CVAE_3D_N64(nn.Module):
    """
    3D Convolutional Variational Autoencoder.

    Parameters
    ==========

    image_channels: number of channels of the image (default=1, gray level dose distribution image)
    input_image_size: the expected shape of each image input
    z_dim: dimension of the latent space (the multinormal dimension)
    """
    def __init__(self, image_channels = 1, z_dim = 32, input_image_size = None):
        super(CVAE_3D_N64, self).__init__()

        # Input size
        assert input_image_size is not None
        self.input_image_size = np.asarray(input_image_size).astype(int)
        assert self.input_image_size.shape == (3,)
        # Encoder part
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=image_channels, out_channels=16, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=16),
            nn.ReLU(),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=32),
            nn.ReLU(),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=64),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=64),
            nn.ReLU(),
            nn.Flatten() # reshape layer
        )

        # Encode to the latent space of normal distributions
        # z_dim is set by user
        # h_dim is the dimension of the encoder's last layer
        # mu_x = encoder(fc1(x)) and sigma_x = encoder(fc2(x))
        n_channels_end = 64
        n_conv3d_layers = 4
        output_image_shape = [int(v) for v in self.input_image_size - n_conv3d_layers*3]
        h_dim = n_channels_end * np.prod(output_image_shape) # - nb_conv3D*(kernel_size-1)
        self.h_dim = h_dim
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        # fc3(z ~ N(mu_x, sigma_x)) => decoder
        self.fc3 = nn.Linear(z_dim, h_dim)

        # Decoder part
        unflatten_size = [n_channels_end] + output_image_shape
        self.unflatten_size = unflatten_size
        self.decoder = nn.Sequential(
            nn.Unflatten(1, unflatten_size),
            nn.BatchNorm3d(num_features=64),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=64),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=32),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=16),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=16, out_channels=image_channels, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=image_channels),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        # z = mu + std * eps
        z = eps.mul(std).add_(mu)
        return z

    def bottleneck(self, h):
        # print("[INFO] bottleneck h size:", h.size())
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        # print("[INFO] Input data shape:", x.size())
        h = self.encoder(x)
        # print("[INFO] h size:", h.size(), " estimated h size:", self.h_dim)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.decoder(z)
        return z

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        # Sending data to appropriate device
        device = next(self.parameters()).device
        x = x.to(device)

        # Step 1: compute representation (fetch it separately for later clustering)
        z_representation = self.representation(x)

        # Step 2: call full CVAE --> encode & decode
        z, mu, logvar = self.encode(x)
        z = self.fc3(z)
        x_hat = self.decode(z)

        return x_hat, mu, logvar, z_representation

