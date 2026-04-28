document.addEventListener('DOMContentLoaded', () => {
    // Navigation
    const navDashboard = document.getElementById('nav-dashboard');
    const navModels = document.getElementById('nav-models');
    const viewDashboard = document.getElementById('view-dashboard');
    const viewModels = document.getElementById('view-models');

    navDashboard.addEventListener('click', (e) => {
        e.preventDefault();
        viewDashboard.classList.remove('hidden');
        viewModels.classList.add('hidden');
        navDashboard.classList.add('active');
        navModels.classList.remove('active');
    });

    navModels.addEventListener('click', (e) => {
        e.preventDefault();
        viewModels.classList.remove('hidden');
        viewDashboard.classList.add('hidden');
        navModels.classList.add('active');
        navDashboard.classList.remove('active');
        loadModelStats();
    });

    document.getElementById('refresh-model-btn').addEventListener('click', loadModelStats);

    // Dashboard Elements
    const userSelect = document.getElementById('user-select');
    const analyzeBtn = document.getElementById('analyze-btn');
    const resultView = document.getElementById('result-view');
    const loadingState = document.getElementById('loading');
    const errorState = document.getElementById('error');
    const errorMsg = document.getElementById('error-msg');
    
    // UI Elements for Data
    const predCount = document.getElementById('pred-count');
    const pastTxns = document.getElementById('past-txns');
    const demoGrid = document.getElementById('demo-grid');
    const statsGrid = document.getElementById('stats-grid');
    const predConfidence = document.getElementById('pred-confidence');
    const predMargin = document.getElementById('pred-margin');
    const actualValContainer = document.getElementById('actual-val-container');
    const actualValCount = document.getElementById('actual-val-count');
    
    let chartInstance = null;
    let featureChartInstance = null;

    // Load all users
    fetch('/api/users')
        .then(response => response.json())
        .then(users => {
            userSelect.innerHTML = '<option value="">-- Select a Customer ID --</option>';
            users.forEach((user, index) => {
                const opt = document.createElement('option');
                opt.value = user.UniqueID;
                opt.textContent = `[${index + 1}/${users.length}] Customer ${user.UniqueID.substring(0, 8)}...`;
                userSelect.appendChild(opt);
            });
        })
        .catch(err => {
            showError("Failed to load customer list.");
        });

    analyzeBtn.addEventListener('click', () => {
        const userId = userSelect.value;
        if(!userId) return;

        // Reset UI
        resultView.classList.add('hidden');
        errorState.classList.add('hidden');
        loadingState.classList.remove('hidden');
        actualValContainer.classList.add('hidden');

        fetch(`/api/predict/${encodeURIComponent(userId)}`)
            .then(res => {
                if(!res.ok) {
                    return res.json().then(data => { throw new Error(data.error || 'Server Error') });
                }
                return res.json();
            })
            .then(data => {
                loadingState.classList.add('hidden');
                resultView.classList.remove('hidden');
                
                // Animate Numbers
                animateNumber(predCount, 0, data.prediction, 1500);
                pastTxns.textContent = data.total_past_transactions;
                
                if (data.actual_val !== null && data.actual_val !== undefined) {
                    actualValCount.textContent = data.actual_val;
                    actualValContainer.classList.remove('hidden');
                }
                
                // Confidence bounds
                predConfidence.textContent = data.confidence;
                predMargin.textContent = data.error_margin_absolute;
                
                // Demographics & Stats
                renderDemographics(data.demographics);
                renderStats(data.advanced_stats);
                
                // Chart
                renderChart(data.history);
            })
            .catch(err => {
                loadingState.classList.add('hidden');
                showError(err.message);
            });
    });

    function showError(msg) {
        errorState.classList.remove('hidden');
        errorMsg.textContent = msg;
    }

    function animateNumber(element, start, end, duration) {
        let startTime = null;
        
        function updateNumber(currentTime) {
            if (!startTime) startTime = currentTime;
            const progress = Math.min((currentTime - startTime) / duration, 1);
            
            const easeOutProgress = progress === 1 ? 1 : 1 - Math.pow(2, -10 * progress);
            const currentVal = Math.floor(easeOutProgress * (end - start) + start);
            
            element.textContent = currentVal;
            
            if (progress < 1) {
                requestAnimationFrame(updateNumber);
            } else {
                element.textContent = end;
            }
        }
        
        requestAnimationFrame(updateNumber);
    }

    function renderDemographics(demo) {
        demoGrid.innerHTML = '';
        const fields = [
            { key: 'IncomeCategory', label: 'Income Segment' },
            { key: 'CustomerStatus', label: 'Account Status' },
            { key: 'OccupationCategory', label: 'Occupation' },
            { key: 'Gender', label: 'Gender' }
        ];

        fields.forEach(f => {
            const val = demo[f.key] || 'N/A';
            const html = `
                <div class="demo-item">
                    <span class="label">${f.label}</span>
                    <span class="value">${val}</span>
                </div>
            `;
            demoGrid.insertAdjacentHTML('beforeend', html);
        });
    }

    function renderStats(stats) {
        if (!stats) return;
        statsGrid.innerHTML = '';
        
        const fields = [
            { key: 'holiday_count', label: 'Holiday Transactions (Nov-Jan)', prefix: '' },
            { key: 'credit_sum', label: 'Total Credits Received', prefix: 'R ' },
            { key: 'debit_sum', label: 'Total Debits Made', prefix: 'R ' }
        ];

        fields.forEach(f => {
            let val = stats[f.key] || 0;
            if (f.key.includes('sum')) {
                val = val.toLocaleString('en-ZA', { maximumFractionDigits: 0 });
            }
            const html = `
                <div class="demo-item">
                    <span class="label">${f.label}</span>
                    <span class="value">${f.prefix}${val}</span>
                </div>
            `;
            statsGrid.insertAdjacentHTML('beforeend', html);
        });
    }

    function renderChart(history) {
        const ctx = document.getElementById('historyChart').getContext('2d');
        if(chartInstance) {
            chartInstance.destroy();
        }
        if(!history || history.length === 0) return;

        history.sort((a,b) => a.Month.localeCompare(b.Month));
        const labels = history.map(d => d.Month);
        const data = history.map(d => d.count);

        chartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Monthly Transactions',
                    data: data,
                    borderColor: '#00b06b',
                    backgroundColor: 'rgba(0, 176, 107, 0.1)',
                    borderWidth: 3,
                    tension: 0.4,
                    fill: true,
                    pointBackgroundColor: '#22d3ee',
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    pointRadius: 4,
                    pointHoverRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#94a3b8' }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: '#94a3b8', maxTicksLimit: 8 }
                    }
                }
            }
        });
    }

    function loadModelStats() {
        const btn = document.getElementById('refresh-model-btn');
        btn.textContent = "Loading...";
        fetch('/api/model_stats')
        .then(res => res.json())
        .then(data => {
            btn.textContent = "Refresh Model Stats";
            if(data.error) throw new Error(data.error);

            document.getElementById('model-type').textContent = data.model_type;
            document.getElementById('model-params').textContent = "Hyperparameters: " + data.hyperparameters;
            document.getElementById('model-rmsle').textContent = data.validation_rmsle.toFixed(4);

            renderFeatureChart(data.top_features);
        }).catch(e => {
            btn.textContent = "Error Loading";
            console.error(e);
        });
    }

    function renderFeatureChart(features) {
        const ctx = document.getElementById('featureChart').getContext('2d');
        if(featureChartInstance) featureChartInstance.destroy();

        const labels = features.map(f => {
            let parts = f.name.split('_');
            if(parts.length > 2) return parts[0] + '_' + parts[1] + '...';
            return f.name;
        });
        const data = features.map(f => (f.importance * 100).toFixed(2));

        featureChartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Importance (%)',
                    data: data,
                    backgroundColor: 'rgba(34, 211, 238, 0.6)',
                    borderColor: '#22d3ee',
                    borderWidth: 1,
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#94a3b8' },
                        title: { display: true, text: 'Importance %', color: '#94a3b8' }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: '#94a3b8' }
                    }
                }
            }
        });
    }
});
