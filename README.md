# ✨ Deep Learning Image Restoration Studio

A state-of-the-art Deep Learning Image Restoration platform featuring 4 model architectures (**Deep 2D CNN**, **Convolutional Autoencoder**, **Restoration GAN**, and **Restormer Transformer**) for restoring damaged, noisy, blurry, and low-light images. 

Powered by a **React.js (Vite)** glassmorphic frontend and a **FastAPI (PyTorch)** inference engine.

---

## 🌟 Key Features

- **4 Deep Learning Model Architectures**:
  - **CNN**: Deep 2D Convolutional Neural Network with spatial residual learning.
  - **Autoencoder**: Encoder-Decoder Convolutional Autoencoder with spatial compression & skip connections.
  - **GAN**: Perceptual Restoration GAN Generator with Dense Residual Blocks (RRDB) and PatchGAN Discriminator.
  - **Transformer**: Restormer Vision Transformer with Multi-Dconv Head Transposed Self-Attention (MDTA) and Gated Feed-Forward Networks (GDFN).
- **Interactive React.js Frontend**:
  - Drag-and-drop test image upload.
  - Model selection (CNN, Autoencoder, GAN, Transformer) and Optimizer options (Adam, AdamW, SGD, RMSprop).
  - Interactive **Before/After Split Comparison Slider** and **Side-by-Side Grid View**.
  - One-click **Download Restored Image** button.
- **Kaggle Multi-GPU Training Notebooks**:
  - Dual Tesla T4 GPU support (`torch.nn.DataParallel`).
  - Auto-discovery dataset resolution for 7 degradation combination presets.
  - Real-time cell timers, tqdm progress bars, and `.zip` results export (`.pth`, `metrics.json`, `curves.png`, `samples.png`).

---

## 📁 Repository Directory Structure

```text
Image Restoration Project/
│
├── 📄 README.md                   # Complete GitHub Documentation
├── 📄 requirements.txt            # Python Dependencies Specification
├── 📄 .gitignore                  # Git Ignore Configuration
│
├── 📂 frontend/                   # React.js Frontend Application
│   ├── dist/                      # Production build bundle
│   ├── public/                    # Static branding assets
│   ├── src/                       # React components & CSS design system
│   │   ├── App.jsx                # Interactive UI Component
│   │   ├── index.css              # Custom Glassmorphic Styles
│   │   └── main.jsx               # React Entry Point
│   ├── index.html                 # Main HTML Template
│   ├── package.json               # Frontend Dependencies & Scripts
│   └── vite.config.js             # Vite Proxy Configuration
│
└── 📂 backend/                    # PyTorch Engine & FastAPI Backend
    ├── 📂 models/                 # PyTorch Model Architectures
    │   ├── cnn_autoencoder.py     # 2D CNN & Autoencoder Models
    │   ├── dae_vae.py             # Denoising VAE Model
    │   ├── restoration_gan.py     # GAN Generator & PatchGAN Discriminator
    │   ├── restormer.py           # Restormer Vision Transformer
    │   └── ddpm.py                # Diffusion Model
    ├── 📂 degradations/           # Noise, Blur & Low-Light Degradation Modules
    ├── 📂 weights/                # Pre-trained Checkpoint Weights (.pth)
    ├── 📂 cnn/                    # CNN Training Notebook & Checkpoint Results
    ├── 📂 autoencoder/            # Autoencoder Training Notebook & Checkpoint Results
    ├── 📂 gan/                    # GAN Training Notebook & Checkpoint Results
    ├── 📂 transformer/            # Transformer Training Notebook & Checkpoint Results
    ├── 📂 notebooks/              # Dataset Preparation Notebooks
    ├── 📂 tests/                  # Automated Test & Benchmark Scripts
    ├── main.py                    # FastAPI Web API Server
    ├── inference.py               # Model Manager & Image Inference Pipeline
    └── metrics.py                 # PSNR & SSIM Metrics Calculation
```

---

## 🚀 Quickstart & Setup Guide

### 1. Prerequisites
- Python 3.10+
- Node.js 18+ (for frontend development)

---

### 2. Backend Setup & Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/image-restoration-project.git
   cd image-restoration-project
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the FastAPI server:
   ```bash
   python -m uvicorn backend.main:app --port 8000 --reload
   ```

---

### 3. Frontend Setup & Run

1. Navigate to the `frontend` folder:
   ```bash
   cd frontend
   ```

2. Install Node.js dependencies:
   ```bash
   npm install
   ```

3. Launch the React development server:
   ```bash
   npm run dev
   ```

4. Open **`http://localhost:3000`** in your browser!

> *Note: In Production Mode, starting `python -m uvicorn backend.main:app --port 8000` serves both the React application and API on `http://127.0.0.1:8000/`.*

---

## 📊 Model Summary & Benchmark Results

| Model Architecture | Key Mechanism | Checkpoint Weight File | PSNR Gain |
|---|---|---|---|
| **Deep 2D CNN** | Full resolution 2D residual convolutions | `backend/cnn/cnn_model.pth` | High |
| **Convolutional Autoencoder** | Spatial compression & bottleneck representation | `backend/autoencoder/autoencoder_model.pth` | High |
| **Restoration GAN** | Dense RRDB Generator + PatchGAN + VGG Loss | `backend/gan/gan_generator.pth` | Perceptual |
| **Restormer Transformer** | MDTA Transposed Self-Attention & GDFN | `backend/transformer/transformer_model.pth` | State-of-the-Art |

---

## 📜 License
This project is licensed under the MIT License.
