/* ═══════════════════════════════════════════════════════════════════
   JavaScript — Dungeon Limo Music Classifier
   ═══════════════════════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {
  initNavbar();
  initUploadZone();
  initAnimations();
  initProgressBars();
  initNavActive();
});

/* ── Navbar ─────────────────────────────────────────────────────────── */
function initNavbar() {
  const navbar = document.getElementById('navbar');
  const toggle = document.getElementById('navToggle');
  const menu = document.getElementById('navMenu');

  // Scroll effect
  if (navbar) {
    window.addEventListener('scroll', () => {
      navbar.classList.toggle('scrolled', window.scrollY > 20);
    }, { passive: true });
  }

  // Mobile toggle
  if (toggle && menu) {
    toggle.addEventListener('click', () => {
      menu.classList.toggle('open');
      const isOpen = menu.classList.contains('open');
      toggle.setAttribute('aria-expanded', isOpen);
      toggle.querySelectorAll('span')[0].style.transform = isOpen ? 'rotate(45deg) translate(5px, 5px)' : '';
      toggle.querySelectorAll('span')[1].style.opacity = isOpen ? '0' : '1';
      toggle.querySelectorAll('span')[2].style.transform = isOpen ? 'rotate(-45deg) translate(5px, -5px)' : '';
    });

    // Close on outside click
    document.addEventListener('click', (e) => {
      if (!navbar.contains(e.target) && menu.classList.contains('open')) {
        menu.classList.remove('open');
        toggle.querySelectorAll('span').forEach(s => s.style.transform = s.style.opacity = '');
      }
    });
  }
}

/* ── Active nav link ────────────────────────────────────────────────── */
function initNavActive() {
  const path = window.location.pathname;
  document.querySelectorAll('.nav-link').forEach(link => {
    const href = link.getAttribute('href');
    if (href === path || (path === '/' && href === '/') || (path !== '/' && href !== '/' && path.startsWith(href))) {
      link.classList.add('active');
    }
  });
}

/* ── Upload Zone ─────────────────────────────────────────────────────── */
function initUploadZone() {
  const zone = document.getElementById('uploadZone');
  const fileInput = document.getElementById('audioFile');
  const filePreview = document.getElementById('filePreview');
  const previewName = document.getElementById('previewName');
  const previewSize = document.getElementById('previewSize');
  const previewExt = document.getElementById('previewExt');
  const form = document.getElementById('classifyForm');
  const loadingOverlay = document.getElementById('loadingOverlay');

  if (!zone || !fileInput) return;

  // Click to open file dialog
  zone.addEventListener('click', (e) => {
    if (!e.target.closest('button')) {
      fileInput.click();
    }
  });

  // Drag & Drop
  ['dragenter', 'dragover'].forEach(evt => {
    zone.addEventListener(evt, (e) => {
      e.preventDefault();
      zone.classList.add('drag-over');
    });
  });

  ['dragleave', 'dragend'].forEach(evt => {
    zone.addEventListener(evt, () => zone.classList.remove('drag-over'));
  });

  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      setFile(files[0]);
    }
  });

  // File input change
  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
      setFile(fileInput.files[0]);
    }
  });

  function setFile(file) {
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    const size = file.size < 1048576
      ? (file.size / 1024).toFixed(1) + ' KB'
      : (file.size / 1048576).toFixed(1) + ' MB';

    if (previewName) previewName.textContent = file.name;
    if (previewSize) previewSize.textContent = size;
    if (previewExt) previewExt.textContent = ext.toUpperCase();
    if (filePreview) filePreview.style.display = 'flex';

    // Create DataTransfer to set file on input
    const dt = new DataTransfer();
    dt.items.add(file);
    fileInput.files = dt.files;

    // Scroll to submit button
    const submitBtn = document.getElementById('submitBtn');
    if (submitBtn) {
      setTimeout(() => submitBtn.scrollIntoView({ behavior: 'smooth', block: 'center' }), 100);
    }
  }

  // Form submit with loading overlay
  if (form && loadingOverlay) {
    form.addEventListener('submit', (e) => {
      if (!fileInput.files.length) {
        e.preventDefault();
        showToast('Pilih file audio terlebih dahulu!', 'error');
        return;
      }
      loadingOverlay.classList.add('active');
      cycleLoadingText();
    });
  }
}

/* Loading text cycle */
function cycleLoadingText() {
  const texts = [
    '🎵 Memuat file audio...',
    '📊 Mengekstrak fitur MFCC...',
    '🔬 Menghitung Chroma & Spectral...',
    '⚡ Menormalisasi fitur...',
    '🤖 SVM sedang mengklasifikasikan...',
    '✨ Menyiapkan hasil...',
  ];
  let i = 0;
  const el = document.getElementById('loadingStep');
  if (!el) return;
  el.textContent = texts[0];
  const interval = setInterval(() => {
    i = (i + 1) % texts.length;
    el.textContent = texts[i];
  }, 1800);
  return interval;
}

/* ── Animate on scroll ───────────────────────────────────────────────── */
function initAnimations() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.style.opacity = '1';
        entry.target.style.transform = 'translateY(0)';
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });

  document.querySelectorAll('.animate-on-scroll').forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(24px)';
    el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    observer.observe(el);
  });
}

/* ── Progress Bars (delayed fill on load) ───────────────────────────── */
function initProgressBars() {
  // Confidence bar
  const confBar = document.getElementById('confidenceBar');
  if (confBar) {
    const target = confBar.dataset.value;
    setTimeout(() => { confBar.style.width = target + '%'; }, 300);
  }

  // Probability bars
  document.querySelectorAll('.prob-bar-fill[data-value]').forEach((bar, i) => {
    const val = bar.dataset.value;
    bar.style.setProperty('--delay', `${i * 0.08}s`);
    setTimeout(() => { bar.style.width = val + '%'; }, 400 + i * 80);
  });

  // Mini bars in table
  document.querySelectorAll('.mini-bar-fill[data-value]').forEach((bar, i) => {
    const val = bar.dataset.value;
    setTimeout(() => { bar.style.width = val + '%'; }, 200 + i * 60);
  });
}

/* ── Toast Notification ──────────────────────────────────────────────── */
function showToast(message, type = 'info') {
  const existing = document.getElementById('toast-container');
  if (existing) existing.remove();

  const colors = {
    error:   { accent: '#f87171', icon: '⚠️', label: 'Perhatian' },
    success: { accent: '#34d399', icon: '✓',  label: 'Berhasil'  },
    info:    { accent: '#60a5fa', icon: 'ℹ',  label: 'Info'      },
  };
  const c = colors[type] || colors.info;

  const toast = document.createElement('div');
  toast.id = 'toast-container';
  toast.style.cssText = `
    position: fixed; bottom: 2rem; right: 2rem; z-index: 9999;
    background: rgba(15, 15, 25, 0.92);
    border: 1px solid rgba(255,255,255,0.08);
    border-left: 3px solid ${c.accent};
    border-radius: 12px;
    padding: 0.875rem 1.25rem;
    display: flex; align-items: center; gap: 0.75rem;
    font-size: 0.875rem; color: #f1f5f9; max-width: 340px;
    backdrop-filter: blur(16px);
    animation: slideInRight 0.3s ease both;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3), 0 0 0 1px rgba(255,255,255,0.04);
  `;
  toast.innerHTML = `
    <span style="font-size:1.1rem; flex-shrink:0; color:${c.accent};">${c.icon}</span>
    <div style="flex:1; min-width:0;">
      <div style="font-weight:700; font-size:0.8rem; color:${c.accent}; letter-spacing:0.05em; text-transform:uppercase; margin-bottom:2px;">${c.label}</div>
      <div style="color:#cbd5e1; font-size:0.85rem; line-height:1.4;">${message}</div>
    </div>
  `;
  document.body.appendChild(toast);
  setTimeout(() => {
    toast.style.transition = 'opacity 0.4s ease, transform 0.4s ease';
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(16px)';
    setTimeout(() => toast.remove(), 400);
  }, 3500);
}

/* ── Chart.js — Confusion Matrix ────────────────────────────────────── */
function initConfusionMatrix(matrixData) {
  const ctx = document.getElementById('confusionCanvas');
  if (!ctx || !matrixData) return;

  const labels = matrixData.labels;
  const matrix = matrixData.matrix;

  // Create flat dataset for heatmap-style scatter
  const chartData = [];
  const backgroundColors = [];
  let maxVal = 0;

  matrix.forEach(row => row.forEach(val => { if (val > maxVal) maxVal = val; }));

  matrix.forEach((row, i) => {
    row.forEach((val, j) => {
      chartData.push({ x: j, y: i, v: val });
      const intensity = val / maxVal;
      const r = Math.round(139 + (0 - 139) * intensity);
      const g = Math.round(92 + (212 - 92) * intensity);
      const b = Math.round(246 + (255 - 246) * intensity);
      backgroundColors.push(`rgba(${r},${g},${b},${0.2 + intensity * 0.8})`);
    });
  });

  // Use Chart.js with custom rendering
  if (typeof Chart === 'undefined') return;

  const chart = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [{
        data: chartData,
        backgroundColor: backgroundColors,
        pointRadius: 0,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      aspectRatio: 1,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: (items) => {
              if (!items.length) return '';
              const { x, y } = items[0].raw;
              return `${labels[y]} → ${labels[x]}`;
            },
            label: (item) => `Jumlah: ${item.raw.v}`,
          }
        }
      },
      scales: {
        x: {
          type: 'linear',
          min: -0.5,
          max: labels.length - 0.5,
          ticks: {
            stepSize: 1,
            callback: (v) => labels[v] || '',
            color: '#9d8ec4',
            font: { size: 10 },
          },
          grid: { color: 'rgba(255,255,255,0.04)' },
          title: { display: true, text: 'Prediksi', color: '#9d8ec4', font: { size: 11 } }
        },
        y: {
          type: 'linear',
          min: -0.5,
          max: labels.length - 0.5,
          ticks: {
            stepSize: 1,
            callback: (v) => labels[v] || '',
            color: '#9d8ec4',
            font: { size: 10 },
          },
          grid: { color: 'rgba(255,255,255,0.04)' },
          title: { display: true, text: 'Aktual', color: '#9d8ec4', font: { size: 11 } }
        }
      }
    },
    plugins: [{
      id: 'heatmap',
      afterDraw(chart) {
        const { ctx: c, scales: { x, y } } = chart;
        const cellW = x.getPixelForValue(1) - x.getPixelForValue(0);
        const cellH = y.getPixelForValue(1) - y.getPixelForValue(0);

        matrix.forEach((row, i) => {
          row.forEach((val, j) => {
            const px = x.getPixelForValue(j);
            const py = y.getPixelForValue(i);
            const intensity = val / maxVal;

            c.save();
            c.fillStyle = `rgba(${Math.round(139 + (0-139)*intensity)},${Math.round(92+(212-92)*intensity)},${Math.round(246+(255-246)*intensity)},${0.15 + intensity*0.75})`;
            c.fillRect(px - cellW/2, py - Math.abs(cellH)/2, cellW, Math.abs(cellH));

            c.fillStyle = intensity > 0.5 ? '#f0e6ff' : '#9d8ec4';
            c.font = `${Math.min(11, Math.abs(cellH)*0.35)}px 'JetBrains Mono', monospace`;
            c.textAlign = 'center';
            c.textBaseline = 'middle';
            c.fillText(val, px, py);
            c.restore();
          });
        });
      }
    }]
  });

  return chart;
}

/* ── Chart.js — Bar chart per-genre metrics ─────────────────────────── */
function initMetricsChart(perClassData) {
  const ctx = document.getElementById('metricsChart');
  if (!ctx || !perClassData) return;
  if (typeof Chart === 'undefined') return;

  const genres = Object.keys(perClassData);
  const precision = genres.map(g => +(perClassData[g].precision * 100).toFixed(1));
  const recall = genres.map(g => +(perClassData[g].recall * 100).toFixed(1));
  const f1 = genres.map(g => +(perClassData[g].f1_score * 100).toFixed(1));

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: genres,
      datasets: [
        {
          label: 'Precision (%)',
          data: precision,
          backgroundColor: 'rgba(139,92,246,0.7)',
          borderColor: 'rgba(139,92,246,1)',
          borderWidth: 1,
          borderRadius: 4,
        },
        {
          label: 'Recall (%)',
          data: recall,
          backgroundColor: 'rgba(0,212,255,0.6)',
          borderColor: 'rgba(0,212,255,1)',
          borderWidth: 1,
          borderRadius: 4,
        },
        {
          label: 'F1-Score (%)',
          data: f1,
          backgroundColor: 'rgba(16,217,160,0.6)',
          borderColor: 'rgba(16,217,160,1)',
          borderWidth: 1,
          borderRadius: 4,
        },
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      aspectRatio: 2,
      plugins: {
        legend: {
          labels: { color: '#9d8ec4', font: { size: 11 }, boxWidth: 14 }
        },
        tooltip: {
          backgroundColor: 'rgba(10,6,20,0.95)',
          borderColor: 'rgba(139,92,246,0.3)',
          borderWidth: 1,
          titleColor: '#f0e6ff',
          bodyColor: '#9d8ec4',
        }
      },
      scales: {
        x: {
          ticks: { color: '#9d8ec4', font: { size: 10 } },
          grid: { color: 'rgba(255,255,255,0.04)' },
        },
        y: {
          min: 50,
          max: 100,
          ticks: {
            color: '#9d8ec4',
            font: { size: 10 },
            callback: v => v + '%'
          },
          grid: { color: 'rgba(255,255,255,0.04)' },
        }
      }
    }
  });
}

/* ── CSS Injection for toast animation ───────────────────────────────── */
const style = document.createElement('style');
style.textContent = `
  @keyframes slideInRight {
    from { opacity: 0; transform: translateX(20px); }
    to { opacity: 1; transform: translateX(0); }
  }
`;
document.head.appendChild(style);

// Export untuk template inline scripts
window.DungeonLimo = { initConfusionMatrix, initMetricsChart };
