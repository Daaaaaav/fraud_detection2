// ========== Loading Bar Functions ==========
function startLoading() {
  const loadingBar = document.getElementById("loading-bar");
  if (loadingBar) {
    loadingBar.style.width = "0%";
    loadingBar.style.display = "block";
    setTimeout(() => (loadingBar.style.width = "100%"), 50);
  }
}

function stopLoading() {
  const loadingBar = document.getElementById("loading-bar");
  if (loadingBar) {
    setTimeout(() => {
      loadingBar.style.display = "none";
      loadingBar.style.width = "0%";
    }, 400);
  }
}

// ========== Spinner Functions ==========
function showSpinner() {
  document.getElementById('spinner')?.style.setProperty('display', 'block');
}

function hideSpinner() {
  document.getElementById('spinner')?.style.setProperty('display', 'none');
}

// ========== Toast Notification ==========
function showToast(message) {
  const toast = document.createElement('div');
  toast.className = 'toast-container';
  toast.innerHTML = `<div class="toast-message">${message}</div>`;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3000);
}

// ========== Tab Switching ==========
function setActiveTab(tabId) {
  document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));

  document.querySelector(`.tab-button[data-tab="${tabId}"]`)?.classList.add('active');
  document.getElementById(tabId)?.classList.add('active');
}

// ========== Load Data into Table ==========
async function loadData() {
  const tableHead = document.getElementById('table-head');
  const tableBody = document.getElementById('table-body');
  startLoading();
  try {
    const response = await fetch('/preprocess', { method: 'POST' });
    const result = await response.json();

    if (Array.isArray(result.sample) && result.sample.length) {
      const keys = Object.keys(result.sample[0]);
      tableHead.innerHTML = `<tr>${keys.map(k => `<th>${k}</th>`).join('')}</tr>`;
      tableBody.innerHTML = result.sample.map(row => 
        `<tr>${keys.map(k => `<td>${row[k]}</td>`).join('')}</tr>`
      ).join('');
    } else {
      tableBody.innerHTML = `<tr><td colspan="100%">No data available</td></tr>`;
    }
    console.log('Preprocessing Info:', result.info);
  } catch (error) {
    console.error('Error loading data:', error);
    tableBody.innerHTML = `<tr><td colspan="100%">Error loading data.</td></tr>`;
  } finally {
    stopLoading();
  }
}

// ========== Chart Management ==========
const charts = { rf: null, iso: null, auto: null };

function renderModelChart(modelKey, stats) {
  const chartIds = { rf: 'chart-rf', iso: 'chart-iso', auto: 'chart-auto' };
  const ctx = document.getElementById(chartIds[modelKey])?.getContext('2d');
  if (!ctx) return;

  const data = {
    labels: ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    datasets: [{
      label: modelKey.toUpperCase(),
      data: [
        stats.accuracy ?? 0,
        stats.precision ?? 0,
        stats.recall ?? 0,
        stats.f1_score ?? 0
      ],
      backgroundColor: ['#4caf50', '#2196f3', '#ffc107', '#e91e63']
    }]
  };

  if (charts[modelKey]) {
    charts[modelKey].data = data;
    charts[modelKey].update();
  } else {
    charts[modelKey] = new Chart(ctx, {
      type: 'bar',
      data,
      options: { responsive: true, maintainAspectRatio: false }
    });
  }
}

function updateChartWithStats(stats) {
  if (!stats?.model) return;
  const modelKey = stats.model.toLowerCase().includes('auto') ? 'auto'
                  : stats.model.toLowerCase().includes('isolation') ? 'iso'
                  : 'rf';
  renderModelChart(modelKey, stats);
}

// ========== Training Functions ==========
async function trainModel(endpoint, modelName, resultId, metricId) {
  startLoading();
  try {
    const res = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: modelName })
    });
    const data = await res.json();

    if (resultId) document.getElementById(resultId).textContent = JSON.stringify(data, null, 2);
    if (metricId && data) {
      document.getElementById(metricId).textContent = `
        Model: ${data.model || modelName}
        Accuracy: ${(data.accuracy || 0).toFixed(2)}%
        Precision: ${(data.precision || 0).toFixed(2)}
        Recall: ${(data.recall || 0).toFixed(2)}
        F1 Score: ${(data.f1_score || 0).toFixed(2)}
        Train Samples: ${data.train_samples}
        Test Samples: ${data.test_samples}
      `.trim();
    }

    showToast(data.message || `${modelName} trained.`);
    updateChartWithStats(data);
  } catch (error) {
    console.error(`Error training ${modelName}:`, error);
    showToast(`Failed to train ${modelName}.`);
  } finally {
    stopLoading();
  }
}

const trainRF = () => trainModel('/train/randomforest', 'rf_model', 'trainRFResult', 'rf-metrics');
const trainISO = () => trainModel('/train/isolationforest', 'iso_model', 'trainISOResult');
const trainCombined = () => trainModel('/train/combined', 'rf_model.pkl', 'trainCombinedResult');

// ========== Evaluation Functions ==========
async function evaluateModel(endpoint, resultId, modelName) {
  startLoading();
  try {
    const res = await fetch(endpoint);
    const data = await res.json();
    const { predictions, stats } = data;

    if (Array.isArray(predictions) && predictions.length) {
      renderModelTable(predictions, 'model-eval-head', 'model-eval-body');
      document.getElementById(resultId).textContent = `${modelName} evaluated on ${predictions.length} samples.`;
      updateChartWithStats(stats);
    } else {
      document.getElementById(resultId).textContent = `No evaluation data available for ${modelName}.`;
      showToast(`No evaluation data for ${modelName}.`);
    }
    console.log(`${modelName} Evaluation Stats:`, stats);
  } catch (error) {
    console.error(`Error evaluating ${modelName}:`, error);
    showToast(`Failed to evaluate ${modelName}.`);
  } finally {
    stopLoading();
  }
}

const evaluateRF = () => evaluateModel('/predict/randomforest/all', 'rfEvalResult', 'Random Forest');
const evaluateISO = () => evaluateModel('/predict/isolationforest/all', 'isoEvalResult', 'Isolation Forest');

async function evaluateCombined() {
  try {
    showSpinner();
    const response = await fetch('/predict/combined');
    const { stats } = await response.json();
    document.getElementById('combinedEvalResult').textContent = JSON.stringify(stats, null, 2);
  } catch (error) {
    console.error('Error evaluating combined model:', error);
    alert('Error evaluating combined model.');
  } finally {
    hideSpinner();
  }
}

// ========== Utility Function ==========
function renderModelTable(predictions, headId, bodyId) {
  const tableHead = document.getElementById(headId);
  const tableBody = document.getElementById(bodyId);
  if (!predictions.length) return;

  const keys = Object.keys(predictions[0]);
  tableHead.innerHTML = `<tr>${keys.map(k => `<th>${k}</th>`).join('')}</tr>`;
  tableBody.innerHTML = predictions.map(row =>
    `<tr>${keys.map(k => `<td>${row[k]}</td>`).join('')}</tr>`
  ).join('');
}

// ========== Initialization ==========
document.querySelectorAll('.tab-button').forEach(button =>
  button.addEventListener('click', () => setActiveTab(button.dataset.tab))
);

const toggleInput = document.getElementById('toggle-input');
if (toggleInput) {
  if (localStorage.getItem('darkMode') === 'true') {
    document.body.classList.add('dark');
    toggleInput.checked = true;
  }
  toggleInput.addEventListener('change', () => {
    document.body.classList.toggle('dark');
    localStorage.setItem('darkMode', document.body.classList.contains('dark'));
    showToast('Toggled dark mode');
  });
}

setActiveTab('dataset');
