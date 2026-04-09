// ───────────────────────────────────────────────
//  Food Safety AI  –  Frontend Logic
// ───────────────────────────────────────────────

const dropZone      = document.getElementById('drop-zone');
const fileInput     = document.getElementById('file-input');
const uploadBtn     = document.getElementById('upload-btn');
const previewWrap   = document.getElementById('preview-wrap');
const previewImg    = document.getElementById('preview-img');
const analyzeBtn    = document.getElementById('analyze-btn');
const resultsCard   = document.getElementById('results-card');
const statusBadge   = document.getElementById('status-badge');
const statusText    = document.getElementById('status-text');
const confidenceBar = document.getElementById('confidence-bar');
const confidencePct = document.getElementById('confidence-pct');
const tipIcon       = document.getElementById('tip-icon');
const tipTitle      = document.getElementById('tip-title');
const tipText       = document.getElementById('tip-text');
const loadingOverlay= document.getElementById('loading-overlay');
const scanBtn       = document.getElementById('scan-btn');

// ── Camera Elements ────────────────────────────
const toggleCamBtn    = document.getElementById('toggle-camera-btn');
const cameraContainer = document.getElementById('camera-container');
const cameraFeed      = document.getElementById('camera-feed');
const cameraOverlay   = document.getElementById('camera-result-overlay');
const scanNowBtn      = document.getElementById('scan-now-btn');
const captureCanvas   = document.getElementById('capture-canvas');

let selectedFile = null;
let localStream  = null;

// -- Startup Health Check --
fetch('/api/health')
  .then(r => r.json())
  .then(d => console.log("[INFO] Food Safety AI Backend:", d))
  .catch(e => console.error("[WARN] Could not reach backend health endpoint."));

// ── Drag & Drop ────────────────────────────────
dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) handleFile(file);
});

// ── Click to Upload ────────────────────────────
uploadBtn.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

// ── File Handler ───────────────────────────────
function handleFile(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
    previewWrap.classList.remove('hidden');
    resultsCard.classList.add('hidden');
    scanBtn.disabled = false;
    scanBtn.classList.remove('opacity-50', 'cursor-not-allowed');

    // animate in
    previewWrap.style.opacity = '0';
    previewWrap.style.transform = 'translateY(20px)';
    requestAnimationFrame(() => {
      previewWrap.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
      previewWrap.style.opacity = '1';
      previewWrap.style.transform = 'translateY(0)';
    });
  };
  reader.readAsDataURL(file);
}

// ── Analyze Button ─────────────────────────────
scanBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  await runPrediction(selectedFile);
});

// ── Camera Logic ───────────────────────────────
toggleCamBtn.addEventListener('click', async () => {
  if (localStream) {
    // Stop camera
    localStream.getTracks().forEach(track => track.stop());
    localStream = null;
    cameraContainer.style.display = 'none';
    scanNowBtn.classList.add('hidden');
    toggleCamBtn.textContent = '📷 Open Live Camera';
    cameraOverlay.style.opacity = '0';
  } else {
    try {
      // Check if context is secure
      if (!window.isSecureContext) {
        throw new Error("Camera only works over HTTPS or on localhost. Please check your connection.");
      }

      localStream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          facingMode: 'environment',
          width: { ideal: 1280 },
          height: { ideal: 720 }
        } 
      });
      
      cameraFeed.srcObject = localStream;
      cameraContainer.style.display = 'block';
      cameraContainer.classList.remove('hidden');
      scanNowBtn.classList.remove('hidden');
      toggleCamBtn.textContent = '🛑 Close Camera';
      toggleCamBtn.classList.replace('btn-outline', 'btn-primary');
      cameraOverlay.style.opacity = '0';
      
      // Ensure video plays
      cameraFeed.onloadedmetadata = () => {
        cameraFeed.play();
      };

    } catch (err) {
      console.error("Camera Access Error:", err);
      let msg = err.message;
      if (err.name === 'NotAllowedError') msg = "Permission denied. Please allow camera access in your browser settings.";
      if (err.name === 'NotFoundError') msg = "No camera found on this device.";
      alert("Unable to access camera: " + msg);
    }
  }
});

scanNowBtn.addEventListener('click', async () => {
  if (!localStream) return;
  
  // 1. Capture frame to invisible canvas
  captureCanvas.width = cameraFeed.videoWidth;
  captureCanvas.height = cameraFeed.videoHeight;
  const ctx = captureCanvas.getContext('2d');
  ctx.drawImage(cameraFeed, 0, 0, captureCanvas.width, captureCanvas.height);
  
  // 2. Show scanning overlay immediately over the video feed
  cameraOverlay.textContent = 'SCANNING...';
  cameraOverlay.className = 'camera-overlay-text camera-overlay-scanning';
  cameraOverlay.style.opacity = '1';
  
  // 3. Convert frame to Blob and send
  captureCanvas.toBlob(async (blob) => {
    if (!blob) return;
    const file = new File([blob], "camera_capture.jpg", { type: "image/jpeg" });
    await runCameraPrediction(file);
  }, 'image/jpeg', 0.9);
});

// ── Core Prediction ────────────────────────────
async function runPrediction(file) {
  loadingOverlay.classList.remove('hidden');
  resultsCard.classList.add('hidden');

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      body: formData
    });

    if (!res.ok) throw new Error(`Server error: ${res.statusText}`);

    const data = await res.json();
    displayResult(data);
  } catch (err) {
    console.error(err);
    showError(err.message);
  } finally {
    loadingOverlay.classList.add('hidden');
  }
}

async function runCameraPrediction(file) {
  const formData = new FormData();
  formData.append('file', file);
  
  try {
    const res = await fetch('/api/predict', { method: 'POST', body: formData });
    if (!res.ok) throw new Error(`Server error: ${res.statusText}`);
    const data = await res.json();
    
    // Update camera overlay
    const isEdible = data.label === 'EDIBLE';
    cameraOverlay.textContent = isEdible ? '[EDIBLE]' : '[NOT EDIBLE]';
    cameraOverlay.className = 'camera-overlay-text ' + (isEdible ? 'camera-overlay-edible' : 'camera-overlay-not-edible');
    cameraOverlay.style.opacity = '1';
    
    // Also display the results card downward
    displayResult(data);
  } catch (err) {
    console.error(err);
    cameraOverlay.textContent = 'ERROR';
    cameraOverlay.className = 'camera-overlay-text camera-overlay-not-edible';
    cameraOverlay.style.opacity = '1';
  }
}

// ── Display Results ────────────────────────────
function displayResult(data) {
  const { label, confidence, tip } = data;
  const isEdible = label === 'EDIBLE';
  const pct = Math.round(confidence * 100);

  // Badge
  statusBadge.className = 'status-badge ' + (isEdible ? 'badge-edible' : 'badge-not-edible');
  statusBadge.textContent = isEdible ? '✔ EDIBLE' : '✘ NOT EDIBLE';

  // Status text
  statusText.textContent = isEdible
    ? 'This food appears to be fresh and safe for consumption.'
    : 'This food appears to be spoiled or unsafe for consumption.';

  // Confidence bar
  const barColor = isEdible ? '#22c55e' : '#ef4444';
  confidenceBar.style.width = '0%';
  confidenceBar.style.background = barColor;
  setTimeout(() => { confidenceBar.style.width = pct + '%'; }, 100);
  confidencePct.textContent = pct + '%';

  // Health tip
  tipIcon.textContent  = isEdible ? '🥗' : '⚠️';
  tipTitle.textContent = isEdible ? 'Health Tip' : 'Safety Warning';
  tipTitle.className   = isEdible ? 'tip-title tip-title-safe' : 'tip-title tip-title-warn';
  tipText.textContent  = tip;

  // Show card with animation
  resultsCard.classList.remove('hidden');
  resultsCard.style.opacity = '0';
  resultsCard.style.transform = 'translateY(30px)';
  requestAnimationFrame(() => {
    resultsCard.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    resultsCard.style.opacity = '1';
    resultsCard.style.transform = 'translateY(0)';
  });

  // scroll into view
  resultsCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── Error Handler ──────────────────────────────
function showError(message) {
  resultsCard.classList.remove('hidden');
  statusBadge.className = 'status-badge badge-not-edible';
  statusBadge.textContent = '⚠ ERROR';
  statusText.textContent = message;
  tipIcon.textContent  = '🔧';
  tipTitle.textContent = 'Something went wrong';
  tipTitle.className   = 'tip-title tip-title-warn';
  tipText.textContent  = 'Please ensure the server is running and try again.';
}
