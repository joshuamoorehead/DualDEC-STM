import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 9, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(9*4*4, 10)  # Adjusted for 17x17 input patches
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 9*4*4),
            nn.Unflatten(1, (9, 4, 4)),
            nn.ConvTranspose2d(9, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=5, stride=2, padding=1),
            # nn.Sigmoid() removed due to vanishing gradient problem
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, return_latent=False):
        latent = self.encode(x)
        decoded = self.decode(latent)
        if return_latent:
            return decoded, latent
        return decoded