import unittest
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import CNNAutoencoder, DenoisingVAE, RestorationGANGenerator, Restormer, DDPM

class TestModels(unittest.TestCase):
    def setUp(self):
        self.x = torch.rand(1, 3, 128, 128)

    def test_cnn_autoencoder_forward(self):
        model = CNNAutoencoder()
        out = model(self.x)
        self.assertEqual(out.shape, (1, 3, 128, 128))
        self.assertTrue(torch.all(out >= 0.0) and torch.all(out <= 1.0))

    def test_dae_vae_forward(self):
        model = DenoisingVAE()
        out = model(self.x)
        self.assertEqual(out.shape, (1, 3, 128, 128))

    def test_restoration_gan_forward(self):
        model = RestorationGANGenerator()
        out = model(self.x)
        self.assertEqual(out.shape, (1, 3, 128, 128))

    def test_restormer_forward(self):
        model = Restormer(dim=16, num_blocks=[1, 1, 1, 1], heads=[1, 1, 2, 4])
        out = model(self.x)
        self.assertEqual(out.shape, (1, 3, 128, 128))

    def test_ddpm_forward(self):
        model = DDPM(timesteps=10)
        t = torch.tensor([5])
        noise_pred = model(self.x, t)
        self.assertEqual(noise_pred.shape, (1, 3, 128, 128))

if __name__ == '__main__':
    unittest.main()
