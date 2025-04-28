// Loading Bar Functions
function startLoading() {
  const loadingBar = document.getElementById("loading-bar");
  if (loadingBar) {
    loadingBar.style.width = "0%";
    loadingBar.style.display = "block";
    setTimeout(() => {
      loadingBar.style.width = "100%";
    }, 50);
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

// Spinner Functions
function showSpinner() {
  const spinner = document.getElementById('spinner');
  if (spinner) spinner.style.display = 'block';
}

function hideSpinner() {
  const spinner = document.getElementById('spinner');
  if (spinner) spinner.style.display = 'none';
}

// Toast Notification
function showToast(message) {
  const toastContainer = document.createElement('div');
  toastContainer.className = 'toast-container';

  const toastMessage = document.createElement('div');
  toastMessage.className = 'toast-message';
  toastMessage.textContent = message;

  toastContainer.appendChild(toastMessage);
  document.body.appendChild(toastContainer);

  setTimeout(() => {
    toastContainer.remove();
  }, 3000);
}

// Tab Switching
function setActiveTab(tabId) {
  document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));

  const activeButton = document.querySelector(`.tab-button[data-tab="${tabId}"]`);
  const activeTab = document.getElementById(tabId);

  if (activeButton) activeButton.classList.add('active');
  if (activeTab) activeTab.classList.add('active');
}

// Chart Management
let charts = {
  rf: null,
  iso: null,
  auto: null
};

function renderModelChart(modelKey, stats) {
  const chartIdMap = {
    rf: 'chart-rf',
    iso: 'chart-iso',
    auto: 'chart-auto'
  };
  const chartId = chartIdMap[modelKey];
  const ctx = document.getElementById(chartId)?.getContext('2d');

  if (!ctx) return;

  const data = {
    labels: ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    datasets: [{
      label: modelKey.toUpperCase(),
      data: [
        stats.accuracy || 0,
        stats.precision || 0,
        stats.recall || 0,
        stats.f1_score || 0
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
      options: {
        responsive: true,
        maintainAspectRatio: false
      }
    });
  }
}

function updateChartWithStats(stats) {
  if (!stats || !stats.model) return;
  const modelKey = stats.model.toLowerCase().includes('auto') ? 'auto'
                  : stats.model.toLowerCase().includes('isolation') ? 'iso'
                  : 'rf';
  renderModelChart(modelKey, stats);
}

// Data Loading
async function loadData() {
  const tableHead = document.getElementById('table-head');
  const tableBody = document.getElementById('table-body');
  startLoading();
  try {
    const response = await fetch('/preprocess', { method: 'POST' });
    const result = await response.json();

    if (result.sample && Array.isArray(result.sample)) {
      const keys = Object.keys(result.sample[0]);
      tableHead.innerHTML = '<tr>' + keys.map(k => `<th>${k}</th>`).join('') + '</tr>';
      tableBody.innerHTML = result.sample.map(row =>
        '<tr>' + keys.map(k => `<td>${row[k]}</td>`).join('') + '</tr>'
      ).join('');
    } else {
      tableBody.innerHTML = '<tr><td colspan="100%">No data available</td></tr>';
    }

    console.log('Preprocessing Info:', result.info);
  } catch (error) {
    console.error('Error loading data:', error);
    tableBody.innerHTML = '<tr><td colspan="100%">Error loading data.</td></tr>';
  } finally {
    stopLoading();
  }
}

// Training Functions
async function trainRF() {
  startLoading();
  try {
    const res = await fetch('/train/randomforest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: 'rf_model' })
    });

    const data = await res.json();
    document.getElementById('trainRFResult').textContent = JSON.stringify(data, null, 2);

    if (document.getElementById('rf-metrics')) {
      document.getElementById('rf-metrics').textContent = `
        Model: ${data.model || "Random Forest"}
        Accuracy: ${(data.accuracy || 0).toFixed(2)}%
        Precision: ${(data.precision || 0).toFixed(2)}
        Recall: ${(data.recall || 0).toFixed(2)}
        F1 Score: ${(data.f1_score || 0).toFixed(2)}
        Train Samples: ${data.train_samples}
        Test Samples: ${data.test_samples}
      `.trim();
    }

    showToast(data.message || "Random Forest trained.");
    updateChartWithStats(data);
  } catch (err) {
    console.error("Error training RF:", err);
    showToast("Failed to train Random Forest model.");
  } finally {
    stopLoading();
  }
}

async function trainISO() {
  startLoading();
  try {
    const res = await fetch('/train/isolationforest', { method: 'POST' });
    const data = await res.json();

    document.getElementById('trainISOResult').textContent = JSON.stringify(data, null, 2);
    showToast(data.message);
    updateChartWithStats(data);
  } catch (err) {
    console.error("Error training ISO:", err);
    showToast("Failed to train Isolation Forest model.");
  } finally {
    stopLoading();
  }
}

// Function to Train Combined Model
async function trainCombined(modelName = 'rf_model.pkl') {
  startLoading();
  try {
    const res = await fetch('/train/combined', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: modelName })  // Always send an object
    });

    if (!res.ok) {
      // Optional: handle HTTP errors (e.g., 400, 500)
      throw new Error(`Server error: ${res.status}`);
    }

    const data = await res.json();
    document.getElementById('trainCombinedResult').textContent = JSON.stringify(data, null, 2);

    showToast(data.message || "Combined model trained.");
    updateChartWithStats(data);
  } catch (err) {
    console.error("Error training Combined model:", err);
    showToast("Failed to train Combined model.");
  } finally {
    stopLoading();
  }
}


// Evaluation Functions
async function evaluateCombined() {
  try {
    showSpinner();
    const response = await fetch('/predict/combined');
    const data = await response.json();
    document.getElementById('combinedEvalResult').textContent = JSON.stringify(data.stats, null, 2);
  } catch (error) {
    console.error('Error evaluating combined model:', error);
    alert('Error evaluating combined model.');
  } finally {
    hideSpinner();
  }
}

async function evaluateRF() {
  await evaluateModel('/predict/randomforest/all', 'rfEvalResult', 'Random Forest');
}

async function evaluateISO() {
  await evaluateModel('/predict/isolationforest/all', 'isoEvalResult', 'Isolation Forest');
}

async function evaluateModel(endpoint, resultId, modelName) {
  startLoading();
  try {
    const res = await fetch(endpoint);
    const data = await res.json();

    console.log(`${modelName} Evaluation Response:`, data);
    const predictions = data.predictions;
    const stats = data.stats;
    if (predictions && Array.isArray(predictions) && predictions.length > 0) {
      renderModelTable(predictions, 'model-eval-head', 'model-eval-body');
      document.getElementById(resultId).textContent = `${modelName} evaluated on ${predictions.length} samples.`;
      updateChartWithStats(stats);  
    } else {
      document.getElementById(resultId).textContent = `No evaluation data available for ${modelName}.`;
      showToast(`No evaluation data for ${modelName}.`);
    }

    if (stats) {
      console.log(`${modelName} Stats:`, stats);
    }

  } catch (err) {
    console.error(`Error evaluating ${modelName}:`, err);
    showToast(`Failed to evaluate ${modelName}.`);
  } finally {
    stopLoading();
  }
}


// Initialize
document.querySelectorAll('.tab-button').forEach(button => {
  button.addEventListener('click', () => {
    const targetTabId = button.dataset.tab;
    setActiveTab(targetTabId);
  });
});

setActiveTab("dataset");
