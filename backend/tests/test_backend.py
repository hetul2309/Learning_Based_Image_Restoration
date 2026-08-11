import unittest
from fastapi.testclient import TestClient
import sys
import os
import io
import base64
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.main import app

class TestBackendAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        # Create small test image base64
        img = Image.new('RGB', (64, 64), color='red')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        self.test_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')

    def test_get_models_endpoint(self):
        res = self.client.get('/api/models')
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn('models', data)
        self.assertEqual(len(data['models']), 5)

    def test_degrade_endpoint(self):
        res = self.client.post('/api/degrade', data={
            'image_b64': self.test_b64,
            'noise_type': 'gaussian',
            'noise_severity': '0.1',
            'blur_type': 'gaussian',
            'blur_severity': '0.2',
            'low_light_severity': '0.0'
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn('degraded_b64', data)
        self.assertIn('degraded_metrics', data)

    def test_restore_endpoint(self):
        res = self.client.post('/api/restore', data={
            'model_id': 'cnn_autoencoder',
            'degraded_b64': self.test_b64
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn('restored_b64', data)

    def test_evaluate_endpoint(self):
        res = self.client.post('/api/evaluate', data={
            'clean_b64': self.test_b64,
            'degraded_b64': self.test_b64,
            'restored_b64': self.test_b64
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn('degraded', data)
        self.assertIn('restored', data)
        self.assertIn('gain', data)

if __name__ == '__main__':
    unittest.main()
