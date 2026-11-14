/**
 * frontend/static/js/charts.js
 * ----------------------------
 * Chart.js initialization and dynamic updates for SmartSpam.
 */

let overviewChartInstance = null;
let analyticsChartInstance = null;

const SPAM_COLOR = '#ef4444';
const HAM_COLOR = '#10b981';
const EMPTY_COLOR = '#e2e8f0';

/**
 * Initializes the Overview Doughnut Chart.
 */
function initOverviewChart(spamCount = 0, hamCount = 0) {
  const canvas = document.getElementById('overviewChart');
  if (!canvas) return;

  const total = spamCount + hamCount;
  const isZero = total === 0;

  const chartData = {
    labels: isZero ? ['No Scans Yet'] : ['Spam', 'Legitimate (Ham)'],
    datasets: [{
      data: isZero ? [1] : [spamCount, hamCount],
      backgroundColor: isZero ? [EMPTY_COLOR] : [SPAM_COLOR, HAM_COLOR],
      borderWidth: 2,
      borderColor: '#ffffff',
      hoverOffset: 4
    }]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    cutout: '72%',
    plugins: {
      legend: {
        display: false
      },
      tooltip: {
        enabled: !isZero,
        callbacks: {
          label: function (context) {
            const val = context.raw || 0;
            const pct = total > 0 ? ((val / total) * 100).toFixed(1) : 0;
            return ` ${context.label}: ${val} (${pct}%)`;
          }
        }
      }
    },
    animation: {
      animateScale: true,
      animateRotate: true
    }
  };

  if (overviewChartInstance) {
    overviewChartInstance.destroy();
  }

  overviewChartInstance = new Chart(canvas, {
    type: 'doughnut',
    data: chartData,
    options: chartOptions
  });

  // Update text values
  const spamValElem = document.getElementById('legendSpamVal');
  const hamValElem = document.getElementById('legendHamVal');
  const chartBadgeElem = document.getElementById('chartTotalBadge');

  if (spamValElem) spamValElem.textContent = spamCount;
  if (hamValElem) hamValElem.textContent = hamCount;
  if (chartBadgeElem) chartBadgeElem.textContent = `${total} Total Messages`;
}

/**
 * Initializes the Analytics View Chart.
 */
function initAnalyticsChart(spamCount = 0, hamCount = 0) {
  const canvas = document.getElementById('analyticsDoughnutChart');
  if (!canvas) return;

  const total = spamCount + hamCount;
  const isZero = total === 0;

  const chartData = {
    labels: isZero ? ['No Scans Yet'] : ['Spam', 'Legitimate (Ham)'],
    datasets: [{
      data: isZero ? [1] : [spamCount, hamCount],
      backgroundColor: isZero ? [EMPTY_COLOR] : [SPAM_COLOR, HAM_COLOR],
      borderWidth: 2,
      borderColor: '#ffffff',
      hoverOffset: 6
    }]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    cutout: '68%',
    plugins: {
      legend: {
        position: 'bottom',
        labels: {
          font: { family: "'Inter', sans-serif", size: 12 },
          padding: 16,
          usePointStyle: true
        }
      },
      tooltip: {
        enabled: !isZero
      }
    }
  };

  if (analyticsChartInstance) {
    analyticsChartInstance.destroy();
  }

  analyticsChartInstance = new Chart(canvas, {
    type: 'doughnut',
    data: chartData,
    options: chartOptions
  });
}

/**
 * Updates both charts dynamically with new stats without rebuilding.
 */
function updateCharts(spamCount = 0, hamCount = 0) {
  const total = spamCount + hamCount;
  const isZero = total === 0;

  // Update overview chart
  if (overviewChartInstance) {
    overviewChartInstance.data.labels = isZero ? ['No Scans Yet'] : ['Spam', 'Legitimate (Ham)'];
    overviewChartInstance.data.datasets[0].data = isZero ? [1] : [spamCount, hamCount];
    overviewChartInstance.data.datasets[0].backgroundColor = isZero ? [EMPTY_COLOR] : [SPAM_COLOR, HAM_COLOR];
    overviewChartInstance.options.plugins.tooltip.enabled = !isZero;
    overviewChartInstance.update();

    const spamValElem = document.getElementById('legendSpamVal');
    const hamValElem = document.getElementById('legendHamVal');
    const chartBadgeElem = document.getElementById('chartTotalBadge');

    if (spamValElem) spamValElem.textContent = spamCount;
    if (hamValElem) hamValElem.textContent = hamCount;
    if (chartBadgeElem) chartBadgeElem.textContent = `${total} Total Messages`;
  } else {
    initOverviewChart(spamCount, hamCount);
  }

  // Update analytics chart
  if (analyticsChartInstance) {
    analyticsChartInstance.data.labels = isZero ? ['No Scans Yet'] : ['Spam', 'Legitimate (Ham)'];
    analyticsChartInstance.data.datasets[0].data = isZero ? [1] : [spamCount, hamCount];
    analyticsChartInstance.data.datasets[0].backgroundColor = isZero ? [EMPTY_COLOR] : [SPAM_COLOR, HAM_COLOR];
    analyticsChartInstance.options.plugins.tooltip.enabled = !isZero;
    analyticsChartInstance.update();
  } else {
    initAnalyticsChart(spamCount, hamCount);
  }
}

