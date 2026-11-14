/**
 * frontend/static/js/app.js
 * -------------------------
 * Core Client Application Logic for SmartSpam.
 * Manages view routing, real-time message classification,
 * database audit log operations, and live dashboard metrics.
 */

document.addEventListener('DOMContentLoaded', () => {
  // =========================================================================
  // State & Preset Data
  // =========================================================================
  const sampleMessages = {
    prize: "WINNER!! As a valued network customer you have won a £1,000 cash prize! Call 09061743806 immediately to claim your award. T&Cs apply.",
    phishing: "URGENT SECURITY ALERT: Your online banking account access has been temporarily restricted. Verify your password and identity at http://secure-auth.bank-login.com to restore access.",
    urgent: "Free entry in a £500 weekly bonus draw! Text CLAIM to 87121 now to claim your code. Standard txt rates apply. Call helpline on 08452810075.",
    meeting: "Hey Alex, are we still meeting for lunch at 1 PM today? Let me know if you want me to reserve a table at the cafe.",
    academic: "Hi Professor, could you please share the reference slides and dataset link from yesterday's Natural Language Processing lecture? Thank you!"
  };

  const viewTitles = {
    overview: {
      title: "Dashboard Overview",
      subtitle: "Real-time NLP message classification and security monitoring."
    },
    detector: {
      title: "Spam Detector",
      subtitle: "Instant SMS and email text analysis with calibrated probability."
    },
    history: {
      title: "Message History & Audit Log",
      subtitle: "Persistent classification records stored in SQLite database."
    },
    analytics: {
      title: "Model Analytics & Benchmarks",
      subtitle: "Multi-algorithm evaluation, test set confusion matrix, and performance metrics."
    },
    about: {
      title: "About SmartSpam Project",
      subtitle: "Architecture, NLP preprocessing, TF-IDF, and machine learning theory."
    }
  };

  // DOM Elements
  const navLinks = document.querySelectorAll('.nav-link');
  const contentViews = document.querySelectorAll('.content-view');
  const pageTitle = document.getElementById('pageTitle');
  const pageSubtitle = document.getElementById('pageSubtitle');
  const sidebar = document.getElementById('sidebar');
  const sidebarBackdrop = document.getElementById('sidebarBackdrop');
  const menuToggleBtn = document.getElementById('menuToggleBtn');
  const mobileCloseBtn = document.getElementById('mobileCloseBtn');
  const headerScanBtn = document.getElementById('headerScanBtn');
  const ctaScanBtn = document.getElementById('ctaScanBtn');
  const emptyScanBtn = document.getElementById('emptyScanBtn');

  // Form Elements
  const spamForm = document.getElementById('spamDetectorForm');
  const messageInput = document.getElementById('messageInput');
  const charCounter = document.getElementById('charCounter');
  const clearBtn = document.getElementById('clearBtn');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const analyzeBtnText = document.getElementById('analyzeBtnText');
  const analyzeSpinner = document.getElementById('analyzeSpinner');
  const samplePills = document.querySelectorAll('.pill-btn');

  // Result Elements
  const resultContainer = document.getElementById('resultContainer');
  const resultCard = document.getElementById('resultCard');
  const resultBadgeText = document.getElementById('resultBadgeText');
  const resultModelTag = document.getElementById('resultModelTag');
  const resultConfidenceNumber = document.getElementById('resultConfidenceNumber');
  const resultProgressFill = document.getElementById('resultProgressFill');
  const resultProbHam = document.getElementById('resultProbHam');
  const resultProbSpam = document.getElementById('resultProbSpam');
  const resultExplanation = document.getElementById('resultExplanation');
  const resultCleanedTokens = document.getElementById('resultCleanedTokens');
  const resultTimestamp = document.getElementById('resultTimestamp');
  const copyResultBtn = document.getElementById('copyResultBtn');

  // History Elements
  const historyTableBody = document.getElementById('historyTableBody');
  const historyEmptyState = document.getElementById('historyEmptyState');
  const historySearchInput = document.getElementById('historySearchInput');
  const refreshHistoryBtn = document.getElementById('refreshHistoryBtn');

  // KPI Elements
  const kpiTotalAnalyzed = document.getElementById('kpiTotalAnalyzed');
  const kpiTotalSpam = document.getElementById('kpiTotalSpam');
  const kpiTotalHam = document.getElementById('kpiTotalHam');
  const kpiSpamRate = document.getElementById('kpiSpamRate');
  const kpiAvgConfidence = document.getElementById('kpiAvgConfidence');
  const overviewRecentList = document.getElementById('overviewRecentList');

  let currentLastResult = null;
  let cachedHistory = [];

  // =========================================================================
  // 1. Navigation & Routing
  // =========================================================================
  function switchView(viewName) {
    if (!viewTitles[viewName]) viewName = 'overview';

    // Update active nav links
    navLinks.forEach(link => {
      if (link.dataset.view === viewName) {
        link.classList.add('active');
      } else {
        link.classList.remove('active');
      }
    });

    // Update visible view
    contentViews.forEach(view => {
      if (view.id === `view-${viewName}`) {
        view.classList.add('active');
      } else {
        view.classList.remove('active');
      }
    });

    // Update header
    if (pageTitle && pageSubtitle) {
      pageTitle.textContent = viewTitles[viewName].title;
      pageSubtitle.textContent = viewTitles[viewName].subtitle;
    }

    // Close mobile sidebar if open
    closeMobileSidebar();

    // Specific view activations
    if (viewName === 'overview' || viewName === 'analytics') {
      loadStats();
    }
    if (viewName === 'history') {
      loadHistory();
    }
  }

  navLinks.forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const viewName = link.dataset.view;
      window.location.hash = viewName;
      switchView(viewName);
    });
  });

  // Shortcut scan buttons
  [headerScanBtn, ctaScanBtn, emptyScanBtn].forEach(btn => {
    if (btn) {
      btn.addEventListener('click', () => {
        window.location.hash = 'detector';
        switchView('detector');
        if (messageInput) messageInput.focus();
      });
    }
  });

  // Mobile sidebar controls
  function openMobileSidebar() {
    sidebar.classList.add('open');
    sidebarBackdrop.classList.add('active');
  }

  function closeMobileSidebar() {
    sidebar.classList.remove('open');
    sidebarBackdrop.classList.remove('active');
  }

  if (menuToggleBtn) menuToggleBtn.addEventListener('click', openMobileSidebar);
  if (mobileCloseBtn) mobileCloseBtn.addEventListener('click', closeMobileSidebar);
  if (sidebarBackdrop) sidebarBackdrop.addEventListener('click', closeMobileSidebar);

  // Read hash on load
  const initialHash = window.location.hash.replace('#', '') || 'overview';
  switchView(initialHash);

  // =========================================================================
  // 2. Character Counter & Sample Presets
  // =========================================================================
  if (messageInput && charCounter) {
    messageInput.addEventListener('input', () => {
      const len = messageInput.value.length;
      charCounter.textContent = `${len} / 5000 characters`;
      if (len > 4800) {
        charCounter.style.color = '#ef4444';
      } else {
        charCounter.style.color = '';
      }
    });
  }

  samplePills.forEach(pill => {
    pill.addEventListener('click', () => {
      const sampleKey = pill.dataset.sample;
      if (sampleMessages[sampleKey] && messageInput) {
        messageInput.value = sampleMessages[sampleKey];
        messageInput.dispatchEvent(new Event('input'));
        messageInput.focus();
        showToast(`Loaded "${pill.textContent.trim()}" sample.`);
      }
    });
  });

  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      if (messageInput) {
        messageInput.value = '';
        messageInput.dispatchEvent(new Event('input'));
        messageInput.focus();
      }
      if (resultContainer) {
        resultContainer.classList.add('hidden');
      }
    });
  }

  // =========================================================================
  // 3. Spam Classification Submission
  // =========================================================================
  if (spamForm) {
    spamForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const text = messageInput.value.trim();

      if (!text) {
        showToast('Please enter message text to analyze.', 'error');
        messageInput.focus();
        return;
      }

      setLoading(true);

      try {
        const response = await fetch('/api/predict', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ message: text })
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
          throw new Error(data.message || 'Classification request failed.');
        }

        renderPredictionResult(data);
        showToast(`Classified as ${data.classification} (${data.confidence}% confidence)`, 'success');

        // Immediately update background stats and history
        loadStats();
      } catch (err) {
        console.error('Prediction error:', err);
        showToast(err.message || 'An error occurred during classification.', 'error');
      } finally {
        setLoading(false);
      }
    });
  }

  function setLoading(isLoading) {
    if (!analyzeBtn) return;
    analyzeBtn.disabled = isLoading;
    if (isLoading) {
      analyzeSpinner.classList.remove('hidden');
      analyzeBtnText.textContent = 'Classifying...';
    } else {
      analyzeSpinner.classList.add('hidden');
      analyzeBtnText.textContent = 'Analyze Message';
    }
  }

  function renderPredictionResult(data) {
    currentLastResult = data;
    const isSpam = data.is_spam;
    const confidence = data.confidence;
    const probSpam = (data.probabilities.spam * 100).toFixed(1);
    const probHam = (data.probabilities.ham * 100).toFixed(1);

    // Update Result Card classes
    resultCard.classList.remove('result-spam', 'result-ham');
    resultCard.classList.add(isSpam ? 'result-spam' : 'result-ham');

    // Update Badge
    resultBadgeText.textContent = isSpam ? 'SPAM DETECTED' : 'LEGITIMATE MESSAGE (HAM)';
    resultModelTag.textContent = data.model_name || 'Calibrated LinearSVC';

    // Update Confidence Number
    resultConfidenceNumber.textContent = `${confidence.toFixed(1)}%`;

    // Update Progress Bar
    resultProgressFill.style.width = '0%';
    setTimeout(() => {
      resultProgressFill.style.width = `${confidence}%`;
    }, 50);

    // Update Probabilities
    resultProbHam.textContent = `P(Ham): ${probHam}%`;
    resultProbSpam.textContent = `P(Spam): ${probSpam}%`;

    // Dynamic Explanation
    if (isSpam) {
      resultExplanation.textContent =
        `The model identified high-confidence spam markers in this message. These commonly include financial reward triggers, urgent action phrases, normalized phone/numeric codes, or suspicious URL tokens that deviate sharply from organic communication patterns.`;
    } else {
      resultExplanation.textContent =
        `The model classified this message as legitimate (Ham). The linguistic distribution features organic conversation markers and lacks the typical unsolicited broadcast patterns, phishing keywords, or pressure triggers associated with spam.`;
    }

    // Normalized NLP Tokens
    resultCleanedTokens.textContent = data.cleaned_text || '<no vocabulary tokens remaining>';

    // Timestamp
    const formattedDate = new Date(data.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    resultTimestamp.textContent = `Logged to database with ID #${data.id} at ${formattedDate}`;

    // Reveal Result Card
    resultContainer.classList.remove('hidden');
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  if (copyResultBtn) {
    copyResultBtn.addEventListener('click', () => {
      if (!currentLastResult) return;
      const copyText = `[SmartSpam Result]\nClassification: ${currentLastResult.classification}\nConfidence: ${currentLastResult.confidence}%\nModel: ${currentLastResult.model_name}\nMessage: "${currentLastResult.message}"`;
      navigator.clipboard.writeText(copyText).then(() => {
        showToast('Result summary copied to clipboard!');
      }).catch(() => {
        showToast('Unable to copy to clipboard.', 'error');
      });
    });
  }

  // =========================================================================
  // 4. Message History Operations
  // =========================================================================
  async function loadHistory() {
    try {
      const response = await fetch('/api/history?limit=100');
      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(data.message || 'Failed to fetch history.');
      }

      cachedHistory = data.history || [];
      renderHistoryTable(cachedHistory);
    } catch (err) {
      console.error('History fetch error:', err);
      showToast('Could not load prediction history.', 'error');
    }
  }

  function renderHistoryTable(records) {
    if (!historyTableBody) return;
    historyTableBody.innerHTML = '';

    if (!records || records.length === 0) {
      if (historyEmptyState) historyEmptyState.classList.remove('hidden');
      return;
    }

    if (historyEmptyState) historyEmptyState.classList.add('hidden');

    records.forEach(row => {
      const tr = document.createElement('tr');
      tr.id = `history-row-${row.id}`;

      const isSpam = row.classification === 'Spam';
      const badgeClass = isSpam ? 'badge-spam' : 'badge-ham';
      const confFormatted = row.confidence ? `${Number(row.confidence).toFixed(1)}%` : 'N/A';

      // Format ISO timestamp
      let timeFormatted = row.timestamp;
      try {
        const d = new Date(row.timestamp);
        timeFormatted = d.toLocaleString([], {
          year: 'numeric',
          month: 'short',
          day: 'numeric',
          hour: '2-digit',
          minute: '2-digit'
        });
      } catch (e) {
        // fallback
      }

      tr.innerHTML = `
        <td style="font-weight: 600; color: #64748b;">#${row.id}</td>
        <td>
          <div class="table-msg" title="${escapeHtml(row.message)}">${escapeHtml(row.message)}</div>
        </td>
        <td>
          <span class="badge ${badgeClass}">${escapeHtml(row.classification)}</span>
        </td>
        <td style="font-weight: 600; color: ${isSpam ? '#ef4444' : '#10b981'};">
          ${confFormatted}
        </td>
        <td class="table-time">${timeFormatted}</td>
        <td style="text-align: center;">
          <button class="btn btn-danger btn-sm" onclick="window.deleteHistoryRecord(${row.id})" title="Delete record">
            Delete
          </button>
        </td>
      `;

      historyTableBody.appendChild(tr);
    });
  }

  // Filter history locally
  if (historySearchInput) {
    historySearchInput.addEventListener('input', (e) => {
      const q = e.target.value.toLowerCase().trim();
      if (!q) {
        renderHistoryTable(cachedHistory);
        return;
      }
      const filtered = cachedHistory.filter(r =>
        r.message.toLowerCase().includes(q) ||
        r.classification.toLowerCase().includes(q) ||
        String(r.id).includes(q)
      );
      renderHistoryTable(filtered);
    });
  }

  if (refreshHistoryBtn) {
    refreshHistoryBtn.addEventListener('click', () => {
      loadHistory();
      showToast('History refreshed.');
    });
  }

  // Expose delete function to window for row onclick
  window.deleteHistoryRecord = async function (recordId) {
    if (!confirm(`Are you sure you want to delete prediction record #${recordId}?`)) {
      return;
    }

    try {
      const response = await fetch(`/api/history/${recordId}`, {
        method: 'DELETE'
      });
      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(data.message || 'Failed to delete record.');
      }

      showToast(`Record #${recordId} deleted.`, 'success');

      // Animate row removal
      const rowElem = document.getElementById(`history-row-${recordId}`);
      if (rowElem) {
        rowElem.style.opacity = '0';
        rowElem.style.transform = 'translateX(20px)';
        rowElem.style.transition = 'all 0.25s ease';
        setTimeout(() => {
          rowElem.remove();
          cachedHistory = cachedHistory.filter(r => r.id !== recordId);
          if (cachedHistory.length === 0 && historyEmptyState) {
            historyEmptyState.classList.remove('hidden');
          }
        }, 250);
      }

      // Refresh Stats
      loadStats();
    } catch (err) {
      console.error('Delete error:', err);
      showToast(err.message || 'Failed to delete record.', 'error');
    }
  };

  // =========================================================================
  // 5. Dashboard Statistics Operations
  // =========================================================================
  async function loadStats() {
    try {
      const response = await fetch('/api/stats');
      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error('Failed to retrieve statistics.');
      }

      const s = data.stats || {};
      const total = s.total_analyzed || 0;
      const spam = s.total_spam || 0;
      const ham = s.total_ham || 0;
      const spamPct = s.spam_percentage ? `${s.spam_percentage.toFixed(1)}%` : '0.0%';
      const avgConf = s.average_confidence ? `${s.average_confidence.toFixed(1)}%` : '0.0%';

      // Update KPI DOM
      if (kpiTotalAnalyzed) animateNumber(kpiTotalAnalyzed, total);
      if (kpiTotalSpam) animateNumber(kpiTotalSpam, spam);
      if (kpiTotalHam) animateNumber(kpiTotalHam, ham);
      if (kpiSpamRate) kpiSpamRate.textContent = spamPct;
      if (kpiAvgConfidence) kpiAvgConfidence.textContent = `Avg Conf: ${avgConf}`;

      // Update Charts via charts.js
      if (typeof updateCharts === 'function') {
        updateCharts(spam, ham);
      }

      // Update Overview recent preview
      loadOverviewRecent();
    } catch (err) {
      console.error('Stats fetch error:', err);
    }
  }

  async function loadOverviewRecent() {
    if (!overviewRecentList) return;
    try {
      const response = await fetch('/api/history?limit=5');
      const data = await response.json();

      if (!response.ok || !data.success) return;

      const records = data.history || [];
      overviewRecentList.innerHTML = '';

      if (records.length === 0) {
        overviewRecentList.innerHTML = `
          <div class="empty-state-mini">
            <p>No activity yet. Scan your first message to see predictions logged here.</p>
          </div>
        `;
        return;
      }

      records.forEach(r => {
        const isSpam = r.classification === 'Spam';
        const badgeClass = isSpam ? 'badge-spam' : 'badge-ham';
        const confText = r.confidence ? `${Number(r.confidence).toFixed(1)}%` : '';

        const item = document.createElement('div');
        item.className = 'recent-item';
        item.innerHTML = `
          <div class="recent-msg" title="${escapeHtml(r.message)}">${escapeHtml(r.message)}</div>
          <div class="recent-meta">
            <span class="badge ${badgeClass}">${r.classification}</span>
            <span class="recent-conf">${confText}</span>
          </div>
        `;
        overviewRecentList.appendChild(item);
      });
    } catch (e) {
      // ignore preview error
    }
  }

  // Smooth counter animation
  function animateNumber(element, target) {
    const current = parseInt(element.textContent, 10) || 0;
    if (current === target) {
      element.textContent = target;
      return;
    }
    const duration = 400;
    const startTime = performance.now();

    function updateCounter(now) {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const val = Math.floor(current + (target - current) * progress);
      element.textContent = val;
      if (progress < 1) {
        requestAnimationFrame(updateCounter);
      } else {
        element.textContent = target;
      }
    }
    requestAnimationFrame(updateCounter);
  }

  // =========================================================================
  // 6. Toast Notification Helper
  // =========================================================================
  function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;

    container.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(10px)';
      toast.style.transition = 'all 0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, 3500);
  }

  function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  // Initial Data Fetch
  loadStats();
});

