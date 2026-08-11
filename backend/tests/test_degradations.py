import unittest
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from degradations import apply_noise, apply_blur, apply_low_light

class TestDegradations(unittest.TestCase):
    def setUp(self):
        self.img = torch.rand(3, 128, 128)

    def test_noise_shapes_and_bounds(self):
        for n_type in ["gaussian", "poisson", "salt_pepper"]:
            noisy = apply_noise(self.img, noise_type=n_type, severity=0.2)
            self.assertEqual(noisy.shape, self.img.shape)
            self.assertTrue(torch.all(noisy >= 0.0) and torch.all(noisy <= 1.0))

    def test_blur_shapes_and_bounds(self):
        for b_type in ["gaussian", "motion", "defocus"]:
            blurred = apply_blur(self.img, blur_type=b_type, severity=0.3)
            self.assertEqual(blurred.shape, self.img.shape)
            self.assertTrue(torch.all(blurred >= 0.0) and torch.all(blurred <= 1.0))

    def test_low_light_shapes_and_bounds(self):
        dark = apply_low_light(self.img, severity=0.5)
        self.assertEqual(dark.shape, self.img.shape)
        self.assertTrue(torch.all(dark >= 0.0) and torch.all(dark <= 1.0))
        # Ensure image is overall darker than original
        self.assertLess(dark.mean().item(), self.img.mean().item())

if __name__ == '__main__':
    unittest.main()
