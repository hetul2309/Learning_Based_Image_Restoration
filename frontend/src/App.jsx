import React, { useState, useEffect, useRef } from 'react';
import './index.css';

// Sample generator for demonstration presets
function createSampleCanvas(type = 'noisy_face') {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 256;
  const ctx = canvas.getContext('2d');

  if (type === 'noisy_face') {
    // Face with synthetic noise/blur background
    const grad = ctx.createLinearGradient(0, 0, 256, 256);
    grad.addColorStop(0, '#78350f');
    grad.addColorStop(0.5, '#92400e');
    grad.addColorStop(1, '#451a03');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 256, 256);

    ctx.fillStyle = '#fed7aa';
    ctx.beginPath();
    ctx.arc(128, 128, 70, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#1e293b';
    ctx.beginPath();
    ctx.arc(105, 110, 10, 0, Math.PI * 2);
    ctx.arc(151, 110, 10, 0, Math.PI * 2);
    ctx.fill();

    ctx.lineWidth = 6;
    ctx.strokeStyle = '#9a3412';
    ctx.beginPath();
    ctx.arc(128, 135, 35, 0.2 * Math.PI, 0.8 * Math.PI);
    ctx.stroke();

    // Add noise overlay
    const imgData = ctx.getImageData(0, 0, 256, 256);
    const data = imgData.data;
    for (let i = 0; i < data.length; i += 4) {
      const noise = (Math.random() - 0.5) * 50;
      data[i]     = Math.min(255, Math.max(0, data[i] + noise));
      data[i + 1] = Math.min(255, Math.max(0, data[i + 1] + noise));
      data[i + 2] = Math.min(255, Math.max(0, data[i + 2] + noise));
    }
    ctx.putImageData(imgData, 0, 0);

  } else if (type === 'blurry_landscape') {
    const grad = ctx.createLinearGradient(0, 0, 0, 256);
    grad.addColorStop(0, '#0369a1');
    grad.addColorStop(0.6, '#0284c7');
    grad.addColorStop(1, '#ca8a04');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 256, 256);

    ctx.fillStyle = '#eab308';
    ctx.beginPath();
    ctx.arc(200, 70, 30, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#14532d';
    ctx.beginPath();
    ctx.moveTo(0, 256);
    ctx.lineTo(80, 120);
    ctx.lineTo(160, 256);
    ctx.fill();

    // Add low contrast/dark tone
    ctx.fillStyle = 'rgba(0, 0, 0, 0.35)';
    ctx.fillRect(0, 0, 256, 256);

  } else {
    // Dark/low-light scene
    ctx.fillStyle = '#090d16';
    ctx.fillRect(0, 0, 256, 256);

    const grad = ctx.createRadialGradient(128, 128, 5, 128, 128, 100);
    grad.addColorStop(0, 'rgba(56, 189, 248, 0.4)');
    grad.addColorStop(1, 'rgba(15, 23, 42, 0.9)');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(128, 128, 90, 0, Math.PI * 2);
    ctx.fill();
  }

  return canvas.toDataURL('image/png');
}

export default function App() {
  const [uploadedB64, setUploadedB64] = useState(null);
  const [restoredB64, setRestoredB64] = useState(null);

  // Purpose/Goal Selector
  const [restorationPurpose, setRestorationPurpose] = useState('general');

  // Model Selector (4 Core Options: CNN, Autoencoder, GAN, Transformer)
  const [selectedModel, setSelectedModel] = useState('cnn');

  // Optimizer Selector
  const [selectedOptimizer, setSelectedOptimizer] = useState('adam');

  const [isProcessing, setIsProcessing] = useState(false);
  const [activeTab, setActiveTab] = useState('split');
  const [splitPos, setSplitPos] = useState(50);
  const [isDragging, setIsDragging] = useState(false);

  const splitRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    // Set default sample image on mount
    const sample = createSampleCanvas('noisy_face');
    setUploadedB64(sample);
  }, []);

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (evt) => {
        setUploadedB64(evt.target.result);
        setRestoredB64(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSampleClick = (type) => {
    const sample = createSampleCanvas(type);
    setUploadedB64(sample);
    setRestoredB64(null);
  };

  // Execution handler: Uploaded Image -> AI Restoration Model -> Restored Output
  const handleProceed = async () => {
    if (!uploadedB64) return;
    setIsProcessing(true);

    try {
      const resFormData = new FormData();
      resFormData.append('model_id', selectedModel);
      resFormData.append('optimizer', selectedOptimizer);
      resFormData.append('purpose', restorationPurpose);
      resFormData.append('image_b64', uploadedB64);

      const resRes = await fetch('/api/restore', { method: 'POST', body: resFormData });
      if (!resRes.ok) throw new Error('Restoration request failed');
      
      const resData = await resRes.json();
      setRestoredB64(resData.restored_b64);

    } catch (err) {
      console.error('Restoration error:', err);
      alert('Could not connect to backend server. Make sure python backend is running on http://127.0.0.1:8000');
    } finally {
      setIsProcessing(false);
    }
  };

  // Download restored image handler
  const handleDownload = () => {
    if (!restoredB64) return;
    const a = document.createElement('a');
    a.href = restoredB64;
    a.download = `restored_image_${selectedModel}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  // Split slider drag handling
  const handleMouseMove = (e) => {
    if (!isDragging || !splitRef.current) return;
    const rect = splitRef.current.getBoundingClientRect();
    let x = e.clientX - rect.left;
    if (x < 0) x = 0;
    if (x > rect.width) x = rect.width;
    setSplitPos((x / rect.width) * 100);
  };

  return (
    <div className="layout-wrapper" onMouseMove={handleMouseMove} onMouseUp={() => setIsDragging(false)}>
      <div className="app-background">
        <div className="glow-orb orb-1"></div>
        <div className="glow-orb orb-2"></div>
      </div>

      {/* Header */}
      <header className="navbar glass-card">
        <div className="logo-group">
          <div className="logo-icon">✨</div>
          <div className="logo-text">
            <h1>IMAGE RESTORATION <span className="gradient-text">STUDIO</span></h1>
            <p className="subtitle">Restore Damaged, Noisy, Blurry & Low-Light Images with Deep Learning</p>
          </div>
        </div>
        <div className="navbar-actions">
          <span className="badge status-badge">
            <span className="pulse-dot"></span> React.js + PyTorch Active
          </span>
        </div>
      </header>

      {/* Main Workspace Grid */}
      <main className="workspace-grid">
        {/* Left Control Panel */}
        <aside className="sidebar-panel glass-card">
          
          {/* 1. Upload Test Image */}
          <section className="panel-section">
            <h2 className="section-title">🖼️ 1. Upload Image to Restore</h2>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileUpload}
              accept="image/*"
              style={{ display: 'none' }}
            />
            <div className="upload-zone" onClick={() => fileInputRef.current?.click()}>
              <div className="upload-icon">☁️</div>
              <p className="upload-prompt">Click to <span className="highlight">Upload Image</span></p>
              <span className="upload-hint">Upload your noisy, blurry, or dark photo</span>
            </div>

            <div className="sample-preset-group">
              <span className="preset-label">Or Pick Sample Image:</span>
              <div className="preset-buttons">
                <button className="btn-preset" onClick={() => handleSampleClick('noisy_face')}>👤 Noisy Face</button>
                <button className="btn-preset" onClick={() => handleSampleClick('blurry_landscape')}>🏔️ Blurry Scenery</button>
                <button className="btn-preset" onClick={() => handleSampleClick('low_light')}>🌙 Low Light</button>
              </div>
            </div>
          </section>

          {/* 2. Restoration Goal / Purpose Selection */}
          <section className="panel-section">
            <h2 className="section-title">🎯 2. Restoration Purpose</h2>
            <select
              className="custom-select"
              value={restorationPurpose}
              onChange={(e) => setRestorationPurpose(e.target.value)}
            >
              <option value="general">General Restoration (Noise + Blur + Low Light)</option>
              <option value="denoising">Remove Noise (Image Denoising)</option>
              <option value="deblurring">Deblur Image (Image Deblurring)</option>
              <option value="low_light">Enhance Low-Light (Illumination Enhancement)</option>
            </select>
          </section>

          {/* 3. Model Selector (4 Core Options) */}
          <section className="panel-section">
            <h2 className="section-title">🧠 3. Select Model</h2>
            <div className="model-selector-grid">
              {[
                {
                  id: 'cnn',
                  name: 'CNN',
                  fullName: 'Deep 2D CNN',
                  desc: 'Deep 2D Convolutional Neural Network with spatial residual learning',
                  weights: 'cnn_model.pth'
                },
                {
                  id: 'autoencoder',
                  name: 'Autoencoder',
                  fullName: 'Convolutional Autoencoder',
                  desc: 'Encoder-Decoder architecture with spatial compression & bottleneck',
                  weights: 'autoencoder_model.pth'
                },
                {
                  id: 'gan',
                  name: 'GAN',
                  fullName: 'Restoration GAN',
                  desc: 'Dense Residual Block Generator trained with PatchGAN & VGG Loss',
                  weights: 'gan_generator.pth'
                },
                {
                  id: 'transformer',
                  name: 'Transformer',
                  fullName: 'Restormer (Transformer)',
                  desc: 'Multi-Dconv Head Transposed Attention (MDTA) Vision Transformer',
                  weights: 'transformer_model.pth'
                }
              ].map(m => (
                <div
                  key={m.id}
                  className={`model-card ${selectedModel === m.id ? 'active' : ''}`}
                  onClick={() => setSelectedModel(m.id)}
                >
                  <div className="model-card-header">
                    <span className="model-name">{m.name}</span>
                    <span className="model-badge">.pth Ready</span>
                  </div>
                  <p className="model-desc">{m.desc}</p>
                </div>
              ))}
            </div>
          </section>

          {/* 4. Optimizer Selector */}
          <section className="panel-section">
            <h2 className="section-title">⚙️ 4. Select Optimizer</h2>
            <select
              className="custom-select"
              value={selectedOptimizer}
              onChange={(e) => setSelectedOptimizer(e.target.value)}
            >
              <option value="adam">Adam (lr = 1e-4, betas = 0.9, 0.999)</option>
              <option value="adamw">AdamW (weight_decay = 1e-2)</option>
              <option value="sgd">SGD (momentum = 0.9, nesterov = True)</option>
              <option value="rmsprop">RMSprop (alpha = 0.99, eps = 1e-8)</option>
            </select>
          </section>

          {/* 5. PROCEED Action Button */}
          <button className="btn-proceed" onClick={handleProceed} disabled={isProcessing}>
            {isProcessing ? '⏳ Restoring Image...' : '🚀 PROCEED (Restore Image)'}
          </button>
        </aside>

        {/* Right Stage: Image Display & Download */}
        <section className="stage-container">
          <div className="viewer-card glass-card">
            <div className="viewer-header">
              <div className="viewer-tabs">
                <button
                  className={`tab-btn ${activeTab === 'split' ? 'active' : ''}`}
                  onClick={() => setActiveTab('split')}
                >
                  🎛️ Interactive Comparison Slider
                </button>
                <button
                  className={`tab-btn ${activeTab === 'grid' ? 'active' : ''}`}
                  onClick={() => setActiveTab('grid')}
                >
                  🔳 Side-by-Side View
                </button>
              </div>

              {/* Download Button */}
              <div className="viewer-actions">
                <span style={{ fontSize: '0.82rem', color: '#94a3b8', marginRight: '0.75rem' }}>
                  Model: <strong style={{ color: '#06b6d4' }}>{selectedModel.toUpperCase()}</strong> | Optimizer: <strong style={{ color: '#8b5cf6' }}>{selectedOptimizer.toUpperCase()}</strong>
                </span>
                {restoredB64 && (
                  <button className="btn-download" onClick={handleDownload}>
                    📥 Download Restored Image
                  </button>
                )}
              </div>
            </div>

            {activeTab === 'split' ? (
              <div className="view-wrapper" ref={splitRef}>
                <div className="split-container">
                  {/* Restored Output Layer Underneath */}
                  <div className="split-layer">
                    <img src={restoredB64 || uploadedB64} alt="Restored Output" />
                    <span className="layer-badge restored-badge">
                      {restoredB64 ? `Restored Output (${selectedModel.toUpperCase()})` : 'AI Restored Preview'}
                    </span>
                  </div>

                  {/* Uploaded Input Layer Overlay */}
                  <div className="split-layer layer-degraded" style={{ width: `${splitPos}%` }}>
                    <img src={uploadedB64} alt="Uploaded Image" />
                    <span className="layer-badge degraded-badge">Original Uploaded Image</span>
                  </div>

                  {/* Draggable Handle */}
                  <div
                    className="split-handle"
                    style={{ left: `${splitPos}%` }}
                    onMouseDown={() => setIsDragging(true)}
                  >
                    <div className="handle-circle">↔</div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="view-wrapper">
                <div className="grid-three" style={{ gridTemplateColumns: '1fr 1fr' }}>
                  <div className="grid-box">
                    <span className="box-badge degraded-badge">1. Original Uploaded Image</span>
                    <img src={uploadedB64} alt="Original Uploaded" />
                  </div>
                  <div className="grid-box">
                    <span className="box-badge restored-badge">2. AI Restored Output ({selectedModel.toUpperCase()})</span>
                    <img src={restoredB64 || uploadedB64} alt="AI Restored" />
                  </div>
                </div>
              </div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
