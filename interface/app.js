// Configurações globais
const CONFIG = {
    API_BASE_URL: 'http://localhost:5000',
    METRICS_API_URL: 'http://localhost:5000/api/metrics',
    WEBSOCKET_URL: 'ws://localhost:5000',
    UPDATE_INTERVAL: 30000, // 30 segundos
    CHART_COLORS: {
        primary: '#2563eb',
        secondary: '#64748b',
        success: '#22c55e',
        warning: '#f59e0b',
        error: '#ef4444',
        accent: '#10b981'
    },
    ENDPOINTS: {
        PREDICT: '/predict',
        PREDICT_BATCH: '/predict/batch',
        MODELS: '/models',
        STATS: '/stats',
        HISTORY: '/history',
        HEALTH: '/health',
        METRICS_SUMMARY: '/api/metrics/summary',
        METRICS_LIVE: '/api/metrics/live',
        METRICS_EXPORT: '/api/metrics/export'
    }
};

// Estado global da aplicação
const AppState = {
    currentSection: 'dashboard',
    theme: localStorage.getItem('theme') || 'light',
    notifications: [],
    isConnected: false,
    socket: null,
    charts: {},
    updateIntervals: []
};

// Classe principal da aplicação
class LotofacilApp {
    constructor() {
        this.init();
    }

    async init() {
        this.setupEventListeners();
        this.initTheme();
        this.initWebSocket();
        await this.loadInitialData();
        this.startPeriodicUpdates();
        this.showSection('dashboard');
        this.setupNumberSelection();
        this.setupConfigForm();
        this.initializeCharts();
        this.generateNumbersGrid();
    }

    setupEventListeners() {
        // Navegação
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const section = link.dataset.section;
                this.showSection(section);
            });
        });

        // Toggle de tema
        document.getElementById('theme-toggle').addEventListener('click', () => {
            this.toggleTheme();
        });

        // Toggle de notificações
        document.getElementById('notifications-toggle').addEventListener('click', () => {
            this.toggleNotifications();
        });

        // Gerar nova predição - removido elemento inexistente
        // document.getElementById('generate-prediction').addEventListener('click', () => {
        //     this.generateNewPrediction();
        // });

        // Formulários de configuração
        const predictionConfigForm = document.getElementById('prediction-config-form');
        if (predictionConfigForm) {
            predictionConfigForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.savePredictionConfig();
            });
        }

        const notificationsConfigForm = document.getElementById('notifications-config-form');
        if (notificationsConfigForm) {
            notificationsConfigForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.saveNotificationConfig();
            });
        }

        // Filtro de período nas estatísticas
        const statsPeriod = document.getElementById('stats-period');
        if (statsPeriod) {
            statsPeriod.addEventListener('change', (e) => {
                this.updateStatistics(e.target.value);
            });
        }

        // Slider de confiança
        const confidenceThreshold = document.getElementById('confidence-threshold');
        const confidenceValue = document.getElementById('confidence-value');
        if (confidenceThreshold && confidenceValue) {
            confidenceThreshold.addEventListener('input', (e) => {
                confidenceValue.textContent = e.target.value + '%';
            });
        }

        // Limpar notificações
        const clearNotifications = document.getElementById('clear-notifications');
        if (clearNotifications) {
            clearNotifications.addEventListener('click', () => {
                this.clearNotifications();
            });
        }

        // Formulário de configuração
        const configForm = document.getElementById('config-form');
        if (configForm) {
            configForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.saveConfiguration();
            });
        }

        // Botões de ação
        const refreshBtn = document.getElementById('refresh-data');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => {
                this.loadInitialData();
                this.updateCharts();
            });
        }

        // Seleção de números
        document.addEventListener('change', (e) => {
            if (e.target.classList.contains('number-checkbox')) {
                this.updateSelectedCount();
            }
        });

        // Formulário de parâmetros de predição
        const parametersForm = document.getElementById('parameters-form');
        if (parametersForm) {
            parametersForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.saveParameters();
            });
        }

        // Botões de ação dos parâmetros
        const resetParamsBtn = document.getElementById('reset-parameters');
        if (resetParamsBtn) {
            resetParamsBtn.addEventListener('click', () => {
                this.resetParameters();
            });
        }

        const loadParamsBtn = document.getElementById('load-parameters');
        if (loadParamsBtn) {
            loadParamsBtn.addEventListener('click', () => {
                this.loadParameters();
            });
        }

        const applyParamsBtn = document.getElementById('apply-parameters');
        if (applyParamsBtn) {
            applyParamsBtn.addEventListener('click', () => {
                this.applyParameters();
            });
        }

        // Sliders de parâmetros com atualização em tempo real
        const confidenceSlider = document.getElementById('confidence-threshold-param');
        if (confidenceSlider) {
            confidenceSlider.addEventListener('input', (e) => {
                document.getElementById('confidence-threshold-value').textContent = e.target.value + '%';
            });
        }

        const depthSlider = document.getElementById('historical-depth');
        if (depthSlider) {
            depthSlider.addEventListener('input', (e) => {
                document.getElementById('historical-depth-value').textContent = e.target.value + ' jogos';
            });
        }

        const iterationsSlider = document.getElementById('max-iterations');
        if (iterationsSlider) {
            iterationsSlider.addEventListener('input', (e) => {
                document.getElementById('max-iterations-value').textContent = e.target.value;
            });
        }

        // Event listeners para notificações
        const notificationFilters = document.querySelectorAll('.notifications-filters select');
        notificationFilters.forEach(filter => {
            filter.addEventListener('change', () => this.filterNotifications());
        });

        // Configurações de notificação
        const notificationToggles = document.querySelectorAll('.toggle-switch');
        notificationToggles.forEach(toggle => {
            toggle.addEventListener('click', () => {
                toggle.classList.toggle('active');
                this.saveNotificationSettings();
            });
        });

        // Paginação de notificações
        const paginationBtns = document.querySelectorAll('.pagination-btn');
        paginationBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                if (!btn.disabled && !btn.classList.contains('active')) {
                    const page = btn.dataset.page;
                    if (page) {
                        this.loadNotifications(parseInt(page));
                    }
                }
            });
        });
        
        // Menu mobile
        const mobileMenuBtn = document.getElementById('mobile-menu-btn');
        const mobileOverlay = document.getElementById('mobile-overlay');
        
        if (mobileMenuBtn) {
            mobileMenuBtn.addEventListener('click', this.toggleMobileMenu.bind(this));
        }
        
        if (mobileOverlay) {
            mobileOverlay.addEventListener('click', this.closeMobileMenu.bind(this));
        }
    }

    initTheme() {
        document.documentElement.setAttribute('data-theme', AppState.theme);
        const themeIcon = document.querySelector('#theme-toggle i');
        themeIcon.className = AppState.theme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
    }

    toggleTheme() {
        AppState.theme = AppState.theme === 'light' ? 'dark' : 'light';
        localStorage.setItem('theme', AppState.theme);
        this.initTheme();
    }

    initWebSocket() {
        try {
            AppState.socket = io(CONFIG.API_BASE_URL);
            
            AppState.socket.on('connect', () => {
                AppState.isConnected = true;
                this.updateSystemStatus('online');
                this.showToast('Conectado ao sistema', 'success');
            });

            AppState.socket.on('disconnect', () => {
                AppState.isConnected = false;
                this.updateSystemStatus('offline');
                this.showToast('Desconectado do sistema', 'warning');
            });

            AppState.socket.on('new_prediction', (data) => {
                this.handleNewPrediction(data);
            });

            AppState.socket.on('new_result', (data) => {
                this.handleNewResult(data);
            });

            AppState.socket.on('performance_alert', (data) => {
                this.handlePerformanceAlert(data);
            });

            AppState.socket.on('system_update', (data) => {
                this.handleSystemUpdate(data);
            });
            
            // Listener para notificações em tempo real
            AppState.socket.on('new_notification', (notification) => {
                this.handleNewNotification(notification);
            });

        } catch (error) {
            console.error('Erro ao conectar WebSocket:', error);
            this.updateSystemStatus('offline');
        }
    }

    async loadInitialData() {
        this.showLoading(true);
        
        try {
            this.showToast('Carregando dados...', 'info');
            
            await Promise.all([
                this.loadDashboardData(),
                this.loadPredictions(),
                this.loadStatistics(),
                this.loadConfiguration(),
                this.loadLiveMetrics()
            ]);
            
            // Atualizar gráficos
        this.updateCharts();
        
        // Carregar estatísticas detalhadas
        this.loadDetailedStats();
            
            this.showToast('Dados carregados com sucesso!', 'success');
        } catch (error) {
            console.error('Erro ao carregar dados iniciais:', error);
            this.showToast('Erro ao carregar dados iniciais', 'error');
        } finally {
            this.showLoading(false);
        }
    }

    async loadDashboardData() {
        try {
            const response = await fetch(`${CONFIG.API_BASE_URL}/api/dashboard`);
            const data = await response.json();
            
            this.updateDashboardStats(data.stats);
            this.updateRecentResults(data.recent_results);
            this.initPerformanceChart(data.performance_data);
            this.initAccuracyChart(data.accuracy_data);
            
        } catch (error) {
            console.error('Erro ao carregar dados do dashboard:', error);
        }
    }

    async loadPredictions() {
        try {
            const response = await fetch(`${CONFIG.API_BASE_URL}/api/predictions/current`);
            const data = await response.json();
            
            this.updateMainPrediction(data.main_prediction);
            this.updateAlternativePredictions(data.alternative_predictions);
            this.updatePredictionAnalysis(data.analysis);
            
        } catch (error) {
            console.error('Erro ao carregar predições:', error);
        }
    }

    async loadStatistics(period = 30) {
        try {
            const response = await fetch(`${CONFIG.API_BASE_URL}/api/statistics?period=${period}`);
            const data = await response.json();
            
            this.updatePerformanceMetrics(data.metrics);
            this.initFrequencyChart(data.frequency_data);
            this.initTrendsChart(data.trends_data);
            this.initModelsComparisonChart(data.models_data);
            
        } catch (error) {
            console.error('Erro ao carregar estatísticas:', error);
        }
    }

    async loadConfiguration() {
        try {
            const response = await fetch(`${CONFIG.API_BASE_URL}/api/config`);
            const data = await response.json();
            
            this.updateConfigurationForm(data);
            
        } catch (error) {
            console.error('Erro ao carregar configurações:', error);
        }
    }

    showSection(sectionName) {
        // Atualizar navegação
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
        });
        document.querySelector(`[data-section="${sectionName}"]`).classList.add('active');

        // Mostrar seção
        document.querySelectorAll('.section').forEach(section => {
            section.classList.remove('active');
        });
        document.getElementById(sectionName).classList.add('active');

        AppState.currentSection = sectionName;

        // Carregar dados específicos da seção se necessário
        if (sectionName === 'statistics') {
            this.loadStatistics();
        } else if (sectionName === 'notifications') {
            this.loadNotifications();
            this.loadNotificationSettings();
        }
    }

    updateDashboardStats(stats) {
        document.getElementById('accuracy-rate').textContent = `${stats.accuracy_rate}%`;
        document.getElementById('total-predictions').textContent = stats.total_predictions;
        document.getElementById('last-update').textContent = this.formatDate(stats.last_update);
        document.getElementById('model-confidence').textContent = `${stats.model_confidence}%`;
    }

    updateRecentResults(results) {
        const tbody = document.querySelector('#recent-results-table tbody');
        tbody.innerHTML = '';

        results.forEach(result => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${result.contest}</td>
                <td>${this.formatDate(result.date)}</td>
                <td>${result.prediction.join(', ')}</td>
                <td>${result.result.join(', ')}</td>
                <td>${result.hits}</td>
                <td>
                    <span class="status-badge ${result.hits >= 11 ? 'success' : result.hits >= 8 ? 'warning' : 'error'}">
                        ${result.hits >= 11 ? 'Excelente' : result.hits >= 8 ? 'Bom' : 'Regular'}
                    </span>
                </td>
            `;
            tbody.appendChild(row);
        });
    }

    initPerformanceChart(data) {
        const ctx = document.getElementById('performance-chart').getContext('2d');
        
        if (AppState.charts.performance) {
            AppState.charts.performance.destroy();
        }

        AppState.charts.performance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.labels,
                datasets: [{
                    label: 'Taxa de Acerto (%)',
                    data: data.accuracy,
                    borderColor: CONFIG.CHART_COLORS.primary,
                    backgroundColor: CONFIG.CHART_COLORS.primary + '20',
                    tension: 0.4
                }, {
                    label: 'Índice de Confiabilidade (%)',
                    data: data.confidence,
                    borderColor: CONFIG.CHART_COLORS.accent,
                    backgroundColor: CONFIG.CHART_COLORS.accent + '20',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });
    }

    initAccuracyChart(data) {
        const ctx = document.getElementById('accuracy-chart').getContext('2d');
        
        if (AppState.charts.accuracy) {
            AppState.charts.accuracy.destroy();
        }

        AppState.charts.accuracy = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['11-15 acertos', '8-10 acertos', '0-7 acertos'],
                datasets: [{
                    data: [data.excellent, data.good, data.regular],
                    backgroundColor: [
                        CONFIG.CHART_COLORS.success,
                        CONFIG.CHART_COLORS.warning,
                        CONFIG.CHART_COLORS.error
                    ]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }

    updateMainPrediction(prediction) {
        const container = document.getElementById('main-prediction-numbers');
        container.innerHTML = '';

        prediction.numbers.forEach(number => {
            const ball = document.createElement('div');
            ball.className = 'number-ball';
            ball.textContent = number;
            container.appendChild(ball);
        });

        document.getElementById('main-confidence').textContent = `${prediction.confidence}%`;
        document.getElementById('main-timestamp').textContent = this.formatDateTime(prediction.timestamp);
    }

    updateAlternativePredictions(predictions) {
        const container = document.getElementById('alternative-predictions-list');
        container.innerHTML = '';

        predictions.forEach((prediction, index) => {
            const card = document.createElement('div');
            card.className = 'prediction-card';
            card.innerHTML = `
                <h4>Alternativa ${index + 1}</h4>
                <div class="numbers-grid">
                    ${prediction.numbers.map(num => `<div class="number-ball">${num}</div>`).join('')}
                </div>
                <div class="prediction-meta">
                    <span class="confidence">Confiança: <strong>${prediction.confidence}%</strong></span>
                </div>
            `;
            container.appendChild(card);
        });
    }

    async generateNewPrediction() {
        this.showLoading(true);
        
        try {
            const selectedNumbers = this.getSelectedNumbers();
            const modelo = document.getElementById('modelo')?.value || 'tensorflow_basico';
            const useCache = document.querySelector('input[name="cache"]')?.checked || false;
            
            if (selectedNumbers.length > 15) {
                this.showToast('Máximo de 15 números permitidos', 'error');
                return;
            }

            const requestData = {
                numeros_base: selectedNumbers,
                modelo: modelo,
                usar_cache: useCache
            };

            this.showToast('Gerando predição...', 'info');
            
            const response = await fetch(`${CONFIG.API_BASE_URL}/api/predictions/generate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Erro na predição');
            }
            
            const data = await response.json();
            
            if (data.success) {
                this.updateMainPrediction(data.prediction);
                this.showToast('Nova predição gerada com sucesso!', 'success');
                
                // Recarregar dados após predição
                setTimeout(() => this.loadInitialData(), 1000);
            } else {
                this.showToast('Erro ao gerar predição: ' + data.error, 'error');
            }
            
        } catch (error) {
            console.error('Erro ao gerar predição:', error);
            this.showToast(`Erro ao gerar predição: ${error.message}`, 'error');
        } finally {
            this.showLoading(false);
        }
    }

    updateSystemStatus(status) {
        const statusDot = document.getElementById('system-status');
        const statusText = document.getElementById('system-status-text');
        
        if (status === 'online') {
            statusDot.style.backgroundColor = CONFIG.CHART_COLORS.success;
            statusText.textContent = 'Sistema Online';
        } else {
            statusDot.style.backgroundColor = CONFIG.CHART_COLORS.error;
            statusText.textContent = 'Sistema Offline';
        }
    }

    showLoading(show) {
        const overlay = document.getElementById('loading-overlay');
        if (show) {
            overlay.classList.add('show');
        } else {
            overlay.classList.remove('show');
        }
    }

    showToast(message, type = 'info') {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast ${type} fade-in`;
        toast.innerHTML = `
            <div class="toast-content">
                <i class="fas fa-${this.getToastIcon(type)}"></i>
                <span>${message}</span>
            </div>
        `;
        
        container.appendChild(toast);
        
        setTimeout(() => {
            toast.remove();
        }, 5000);
    }

    getToastIcon(type) {
        const icons = {
            success: 'check-circle',
            error: 'exclamation-circle',
            warning: 'exclamation-triangle',
            info: 'info-circle'
        };
        return icons[type] || 'info-circle';
    }

    toggleNotifications() {
        const panel = document.getElementById('notifications-panel');
        panel.classList.toggle('open');
    }

    addNotification(notification) {
        AppState.notifications.unshift(notification);
        this.updateNotificationBadge();
        this.renderNotifications();
    }

    updateNotificationBadge() {
        const badge = document.getElementById('notification-count');
        const unreadCount = AppState.notifications.filter(n => !n.read).length;
        badge.textContent = unreadCount;
        badge.style.display = unreadCount > 0 ? 'block' : 'none';
    }

    renderNotifications() {
        const container = document.getElementById('notifications-list');
        container.innerHTML = '';

        if (AppState.notifications.length === 0) {
            container.innerHTML = '<p class="text-center text-muted">Nenhuma notificação</p>';
            return;
        }

        AppState.notifications.forEach(notification => {
            const item = document.createElement('div');
            item.className = `notification-item ${!notification.read ? 'unread' : ''}`;
            item.innerHTML = `
                <div class="notification-content">
                    <h4>${notification.title}</h4>
                    <p>${notification.message}</p>
                    <small>${this.formatDateTime(notification.timestamp)}</small>
                </div>
            `;
            container.appendChild(item);
        });
    }

    clearNotifications() {
        AppState.notifications = [];
        this.updateNotificationBadge();
        this.renderNotifications();
    }

    handleNewPrediction(data) {
        this.addNotification({
            id: Date.now(),
            title: 'Nova Predição Disponível',
            message: `Predição para o concurso ${data.contest} foi gerada`,
            timestamp: new Date(),
            read: false,
            type: 'prediction'
        });
        
        if (AppState.currentSection === 'predictions') {
            this.loadPredictions();
        }
    }

    handleNewResult(data) {
        this.addNotification({
            id: Date.now(),
            title: 'Novo Resultado',
            message: `Resultado do concurso ${data.contest}: ${data.hits} acertos`,
            timestamp: new Date(),
            read: false,
            type: 'result'
        });
        
        if (AppState.currentSection === 'dashboard') {
            this.loadDashboardData();
        }
    }

    handlePerformanceAlert(data) {
        this.addNotification({
            id: Date.now(),
            title: 'Alerta de Performance',
            message: data.message,
            timestamp: new Date(),
            read: false,
            type: 'alert'
        });
        
        this.showToast(data.message, 'warning');
    }

    handleSystemUpdate(data) {
        this.addNotification({
            id: Date.now(),
            title: 'Atualização do Sistema',
            message: data.message,
            timestamp: new Date(),
            read: false,
            type: 'system'
        });
    }

    startPeriodicUpdates() {
        // Atualizar dashboard a cada 30 segundos
        const dashboardInterval = setInterval(() => {
            if (AppState.currentSection === 'dashboard') {
                this.loadDashboardData();
            }
        }, CONFIG.UPDATE_INTERVAL);
        
        AppState.updateIntervals.push(dashboardInterval);
    }

    formatDate(dateString) {
        const date = new Date(dateString);
        return date.toLocaleDateString('pt-BR');
    }

    formatDateTime(dateString) {
        const date = new Date(dateString);
        return date.toLocaleString('pt-BR');
    }

    async savePredictionConfig() {
        const formData = new FormData(document.getElementById('prediction-config-form'));
        const config = Object.fromEntries(formData.entries());
        
        try {
            const response = await fetch(`${CONFIG.API_BASE_URL}/api/config/prediction`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(config)
            });
            
            if (response.ok) {
                this.showToast('Configurações salvas com sucesso!', 'success');
            } else {
                this.showToast('Erro ao salvar configurações', 'error');
            }
        } catch (error) {
            console.error('Erro ao salvar configurações:', error);
            this.showToast('Erro ao salvar configurações', 'error');
        }
    }

    async saveNotificationConfig() {
        const formData = new FormData(document.getElementById('notifications-config-form'));
        const config = Object.fromEntries(formData.entries());
        
        try {
            const response = await fetch(`${CONFIG.API_BASE_URL}/api/config/notifications`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(config)
            });
            
            if (response.ok) {
                this.showToast('Configurações de notificação atualizadas!', 'success');
            } else {
                this.showToast('Erro ao atualizar configurações', 'error');
            }
        } catch (error) {
            console.error('Erro ao salvar configurações:', error);
            this.showToast('Erro ao salvar configurações', 'error');
        }
    }

    updateStatistics(period) {
        this.loadStatistics(period);
    }

    updatePerformanceMetrics(metrics) {
        const container = document.getElementById('performance-metrics');
        container.innerHTML = '';
        
        Object.entries(metrics).forEach(([key, value]) => {
            const item = document.createElement('div');
            item.className = 'metric-item';
            item.innerHTML = `
                <span class="metric-label">${this.formatMetricLabel(key)}:</span>
                <span class="metric-value">${value}</span>
            `;
            container.appendChild(item);
        });
    }

    formatMetricLabel(key) {
        const labels = {
            accuracy: 'Taxa de Acerto',
            precision: 'Precisão',
            recall: 'Recall',
            f1_score: 'F1 Score',
            mae: 'Erro Médio Absoluto',
            rmse: 'Raiz do Erro Quadrático Médio'
        };
        return labels[key] || key;
    }

    initFrequencyChart(data) {
        const ctx = document.getElementById('frequency-chart').getContext('2d');
        
        if (AppState.charts.frequency) {
            AppState.charts.frequency.destroy();
        }

        AppState.charts.frequency = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.numbers,
                datasets: [{
                    label: 'Frequência',
                    data: data.frequencies,
                    backgroundColor: CONFIG.CHART_COLORS.primary
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }

    initTrendsChart(data) {
        const ctx = document.getElementById('trends-chart').getContext('2d');
        
        if (AppState.charts.trends) {
            AppState.charts.trends.destroy();
        }

        AppState.charts.trends = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.labels,
                datasets: data.datasets.map((dataset, index) => ({
                    ...dataset,
                    borderColor: Object.values(CONFIG.CHART_COLORS)[index % Object.keys(CONFIG.CHART_COLORS).length],
                    backgroundColor: Object.values(CONFIG.CHART_COLORS)[index % Object.keys(CONFIG.CHART_COLORS).length] + '20'
                }))
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top'
                    }
                }
            }
        });
    }

    initModelsComparisonChart(data) {
        const ctx = document.getElementById('models-comparison-chart').getContext('2d');
        
        if (AppState.charts.modelsComparison) {
            AppState.charts.modelsComparison.destroy();
        }

        AppState.charts.modelsComparison = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: data.metrics,
                datasets: data.models.map((model, index) => ({
                    label: model.name,
                    data: model.values,
                    borderColor: Object.values(CONFIG.CHART_COLORS)[index % Object.keys(CONFIG.CHART_COLORS).length],
                    backgroundColor: Object.values(CONFIG.CHART_COLORS)[index % Object.keys(CONFIG.CHART_COLORS).length] + '20'
                }))
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top'
                    }
                },
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });
    }

    updateConfigurationForm(config) {
        // Atualizar formulário de predição
        document.getElementById('model-type').value = config.prediction.model_type || 'tensorflow';
        document.getElementById('confidence-threshold').value = config.prediction.confidence_threshold || 0.8;
        document.getElementById('prediction-count').value = config.prediction.prediction_count || 3;
        document.getElementById('analysis-depth').value = config.prediction.analysis_depth || 'standard';
        
        // Atualizar valor do slider
        document.getElementById('confidence-value').textContent = config.prediction.confidence_threshold || 0.8;
        
        // Atualizar formulário de notificações
        document.getElementById('notify-predictions').checked = config.notifications.predictions !== false;
        document.getElementById('notify-results').checked = config.notifications.results !== false;
        document.getElementById('notify-performance').checked = config.notifications.performance === true;
        document.getElementById('notify-system').checked = config.notifications.system !== false;
    }

    // Método para carregar métricas em tempo real
    async loadLiveMetrics() {
        try {
            const response = await fetch(`${CONFIG.METRICS_API_URL}/live`);
            if (!response.ok) throw new Error('Erro ao carregar métricas');
            
            const data = await response.json();
            this.updateLiveMetrics(data.data);
            
        } catch (error) {
            console.error('Erro ao carregar métricas:', error);
        }
    }

    // Método para atualizar métricas em tempo real
    updateLiveMetrics(metrics) {
        // Atualizar precisão atual
        const accuracyElement = document.getElementById('current-accuracy');
        if (accuracyElement && metrics.current_accuracy) {
            accuracyElement.textContent = `${(metrics.current_accuracy * 100).toFixed(1)}%`;
        }
        
        // Atualizar alertas ativos
        const alertsElement = document.getElementById('active-alerts');
        if (alertsElement && metrics.active_alerts) {
            alertsElement.textContent = metrics.active_alerts.length;
        }
    }

    // Configurar seleção de números
    setupNumberSelection() {
        const numberLabels = document.querySelectorAll('.number-label');
        numberLabels.forEach(label => {
            label.addEventListener('click', () => {
                const checkbox = document.getElementById(label.getAttribute('for'));
                const selectedCount = document.querySelectorAll('.number-checkbox:checked').length;
                
                if (!checkbox.checked && selectedCount >= 15) {
                    this.showToast('Máximo de 15 números permitidos', 'warning');
                    return;
                }
                
                checkbox.checked = !checkbox.checked;
                label.classList.toggle('selected', checkbox.checked);
            });
        });
    }

    // Obter números selecionados
    getSelectedNumbers() {
        const selected = [];
        document.querySelectorAll('.number-checkbox:checked').forEach(checkbox => {
            selected.push(parseInt(checkbox.value));
        });
        return selected.sort((a, b) => a - b);
    }

    // Configurar formulário de configurações
    setupConfigForm() {
        // Carregar configurações salvas
        this.loadSavedConfig();
    }

    // Salvar configurações
    saveConfiguration() {
        const config = {
            confidenceThreshold: document.getElementById('confidence-threshold')?.value,
            predictionCount: document.getElementById('prediction-count')?.value,
            refreshInterval: document.getElementById('refresh-interval')?.value,
            autoUpdate: document.getElementById('auto-update')?.checked,
            enableWebSocket: document.getElementById('enable-websocket')?.checked,
            notifications: {
                predictions: document.getElementById('notify-predictions')?.checked,
                alerts: document.getElementById('notify-alerts')?.checked,
                reports: document.getElementById('notify-reports')?.checked,
                errors: document.getElementById('notify-errors')?.checked
            }
        };

        localStorage.setItem('lotofacil_config', JSON.stringify(config));
        this.showToast('Configurações salvas com sucesso!', 'success');
        
        // Aplicar configurações
        this.applyConfiguration(config);
    }

    // Carregar configurações salvas
    loadSavedConfig() {
        const saved = localStorage.getItem('lotofacil_config');
        if (saved) {
            const config = JSON.parse(saved);
            this.applyConfiguration(config);
        }
    }

    // Aplicar configurações
    applyConfiguration(config) {
        if (config.refreshInterval) {
            clearInterval(this.metricsInterval);
            this.metricsInterval = setInterval(() => {
                this.loadLiveMetrics();
            }, config.refreshInterval * 1000);
        }
    }

    // Exportar dados
    async exportData(format) {
        try {
            const response = await fetch(`${CONFIG.METRICS_API_URL}/export?format=${format}`);
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `lotofacil_data.${format}`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);
                this.showToast(`Dados exportados em ${format.toUpperCase()}`, 'success');
            }
        } catch (error) {
            this.showToast('Erro ao exportar dados', 'error');
        }
    }

    // Limpar cache
    async clearCache() {
        try {
            const response = await fetch(`${CONFIG.API_BASE_URL}/cache/clear`, {
                method: 'POST'
            });
            if (response.ok) {
                this.showToast('Cache limpo com sucesso', 'success');
            }
        } catch (error) {
            this.showToast('Erro ao limpar cache', 'error');
        }
    }

    // Gerar grid de números
    generateNumbersGrid() {
        const numbersGrid = document.getElementById('numbers-grid');
        if (!numbersGrid) return;

        numbersGrid.innerHTML = '';
        for (let i = 1; i <= 25; i++) {
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.id = `num-${i}`;
            checkbox.name = 'numeros';
            checkbox.value = i;
            checkbox.className = 'number-checkbox';
            checkbox.style.display = 'none';

            const label = document.createElement('label');
            label.htmlFor = `num-${i}`;
            label.className = 'number-label';
            label.textContent = i;

            numbersGrid.appendChild(checkbox);
            numbersGrid.appendChild(label);
        }
    }

    // Inicializar gráficos
    initializeCharts() {
        this.charts = {};
        this.createAccuracyChart();
        this.createFrequencyChart();
        this.createTrendsChart();
    }

    // Criar gráfico de precisão por modelo
    createAccuracyChart() {
        const ctx = document.getElementById('accuracy-chart');
        if (!ctx) return;

        this.charts.accuracy = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['TensorFlow Básico', 'Algoritmos Avançados', 'Análise de Padrões'],
                datasets: [{
                    label: 'Precisão (%)',
                    data: [0, 0, 0],
                    backgroundColor: [
                        'rgba(54, 162, 235, 0.8)',
                        'rgba(255, 99, 132, 0.8)',
                        'rgba(75, 192, 192, 0.8)'
                    ],
                    borderColor: [
                        'rgba(54, 162, 235, 1)',
                        'rgba(255, 99, 132, 1)',
                        'rgba(75, 192, 192, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Precisão por Modelo'
                    },
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                }
            }
        });
    }

    // Criar gráfico de frequência dos números
    createFrequencyChart() {
        const ctx = document.getElementById('frequency-chart');
        if (!ctx) return;

        const labels = Array.from({length: 25}, (_, i) => i + 1);
        const data = new Array(25).fill(0);

        this.charts.frequency = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Frequência',
                    data: data,
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Frequência dos Números'
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Números'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Frequência'
                        },
                        beginAtZero: true
                    }
                }
            }
        });
    }

    // Criar gráfico de tendências temporais
    createTrendsChart() {
        const ctx = document.getElementById('trends-chart');
        if (!ctx) return;

        this.charts.trends = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Precisão Geral',
                    data: [],
                    borderColor: 'rgba(54, 162, 235, 1)',
                    backgroundColor: 'rgba(54, 162, 235, 0.1)',
                    tension: 0.4
                }, {
                    label: 'Taxa de Acerto',
                    data: [],
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Tendências Temporais'
                    }
                },
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'day'
                        },
                        title: {
                            display: true,
                            text: 'Data'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Percentual (%)'
                        },
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });
    }

    // Atualizar gráficos com dados reais
    async updateCharts() {
        try {
            // Carregar dados de métricas
            const metricsResponse = await fetch(`${CONFIG.METRICS_API_URL}/summary`);
            if (metricsResponse.ok) {
                const metrics = await metricsResponse.json();
                this.updateChartsWithData(metrics);
            }

            // Carregar dados de frequência
            const statsResponse = await fetch(`${CONFIG.API_BASE_URL}/stats`);
            if (statsResponse.ok) {
                const stats = await statsResponse.json();
                this.updateFrequencyChart(stats);
            }
        } catch (error) {
            console.error('Erro ao atualizar gráficos:', error);
        }
    }

    // Atualizar gráficos com dados
    updateChartsWithData(metrics) {
        // Atualizar gráfico de precisão
        if (this.charts.accuracy && metrics.model_accuracy) {
            const accuracyData = [
                metrics.model_accuracy.tensorflow_basico || 0,
                metrics.model_accuracy.algoritmos_avancados || 0,
                metrics.model_accuracy.analise_padroes || 0
            ];
            this.charts.accuracy.data.datasets[0].data = accuracyData;
            this.charts.accuracy.update();
        }

        // Atualizar gráfico de tendências
        if (this.charts.trends && metrics.historical_data) {
            const labels = metrics.historical_data.map(d => new Date(d.date));
            const precisionData = metrics.historical_data.map(d => d.precision);
            const accuracyData = metrics.historical_data.map(d => d.accuracy);

            this.charts.trends.data.labels = labels;
            this.charts.trends.data.datasets[0].data = precisionData;
            this.charts.trends.data.datasets[1].data = accuracyData;
            this.charts.trends.update();
        }
    }

    // Atualizar gráfico de frequência
    updateFrequencyChart(stats) {
        if (this.charts.frequency && stats.number_frequency) {
            const frequencyData = Array.from({length: 25}, (_, i) => {
                const number = i + 1;
                return stats.number_frequency[number] || 0;
            });
            
            this.charts.frequency.data.datasets[0].data = frequencyData;
            this.charts.frequency.update();
        }
    }

    // Atualizar contador de números selecionados
    updateSelectedCount() {
        const selectedCheckboxes = document.querySelectorAll('.number-checkbox:checked');
        const countElement = document.getElementById('selected-count');
        if (countElement) {
            countElement.textContent = selectedCheckboxes.length;
        }

        // Limitar seleção a 15 números
        if (selectedCheckboxes.length >= 15) {
            document.querySelectorAll('.number-checkbox:not(:checked)').forEach(checkbox => {
                checkbox.disabled = true;
            });
        } else {
            document.querySelectorAll('.number-checkbox').forEach(checkbox => {
                checkbox.disabled = false;
            });
        }
    }

    // Carregar estatísticas detalhadas
    async loadDetailedStats() {
        try {
            // Carregar performance dos modelos
            await this.loadModelPerformance();
            
            // Carregar histórico de predições
            await this.loadPredictionHistory();
            
            // Carregar análise de números
            await this.loadNumberAnalysis();
        } catch (error) {
            console.error('Erro ao carregar estatísticas detalhadas:', error);
            this.showToast('Erro ao carregar estatísticas detalhadas', 'error');
        }
    }

    // Carregar performance dos modelos
    async loadModelPerformance() {
        const tbody = document.querySelector('#model-performance-table tbody');
        if (!tbody) return;

        try {
            const response = await fetch(`${CONFIG.API_BASE_URL}/api/models`);
            const models = await response.json();
            
            tbody.innerHTML = '';
            
            models.forEach(model => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${model.name}</td>
                    <td><span class="status-badge ${model.status}">${model.status}</span></td>
                    <td>${(model.accuracy * 100).toFixed(1)}%</td>
                    <td>${model.predictions_count}</td>
                    <td>${model.last_training || 'N/A'}</td>
                    <td>${model.version}</td>
                `;
                tbody.appendChild(row);
            });
        } catch (error) {
            tbody.innerHTML = '<tr><td colspan="6" class="loading-row">Erro ao carregar dados</td></tr>';
        }
    }

    // Carregar histórico de predições
    async loadPredictionHistory() {
        const tbody = document.querySelector('#prediction-history-table tbody');
        if (!tbody) return;

        try {
            const response = await fetch(`${CONFIG.API_BASE_URL}/api/history?limit=10`);
            const history = await response.json();
            
            tbody.innerHTML = '';
            
            history.forEach(prediction => {
                const row = document.createElement('tr');
                const numbersStr = prediction.numbers ? prediction.numbers.join(', ') : 'N/A';
                const confidenceStr = prediction.confidence ? `${(prediction.confidence * 100).toFixed(1)}%` : 'N/A';
                
                row.innerHTML = `
                    <td>${new Date(prediction.timestamp).toLocaleString('pt-BR')}</td>
                    <td>${prediction.model}</td>
                    <td>${numbersStr}</td>
                    <td>${confidenceStr}</td>
                    <td><span class="status-badge ${prediction.result || 'pending'}">${prediction.result || 'Pendente'}</span></td>
                `;
                tbody.appendChild(row);
            });
        } catch (error) {
            tbody.innerHTML = '<tr><td colspan="5" class="loading-row">Erro ao carregar histórico</td></tr>';
        }
    }

    // Carregar análise de números
    async loadNumberAnalysis() {
        try {
            const response = await fetch(`${CONFIG.API_BASE_URL}/api/stats`);
            const stats = await response.json();
            
            // Números mais frequentes
            const frequentElement = document.getElementById('frequent-numbers');
            if (frequentElement && stats.most_frequent) {
                frequentElement.innerHTML = stats.most_frequent.slice(0, 10).map(num => 
                    `<span class="number-badge frequent">${num}</span>`
                ).join('');
            }
            
            // Números menos frequentes
            const rareElement = document.getElementById('rare-numbers');
            if (rareElement && stats.least_frequent) {
                rareElement.innerHTML = stats.least_frequent.slice(0, 10).map(num => 
                    `<span class="number-badge rare">${num}</span>`
                ).join('');
            }
            
            // Números em tendência
            const trendingElement = document.getElementById('trending-numbers');
            if (trendingElement && stats.trending) {
                trendingElement.innerHTML = stats.trending.slice(0, 10).map(num => 
                    `<span class="number-badge trending">${num}</span>`
                ).join('');
            }
        } catch (error) {
            console.error('Erro ao carregar análise de números:', error);
            
            // Mostrar mensagem de erro nos elementos
            ['frequent-numbers', 'rare-numbers', 'trending-numbers'].forEach(id => {
                const element = document.getElementById(id);
                if (element) {
                    element.innerHTML = '<span class="loading-text">Erro ao carregar dados</span>';
                }
            });
        }
    }

    // === MÉTODOS DE PARÂMETROS DE PREDIÇÃO ===
    
    // Salvar parâmetros de predição
    async saveParameters() {
        try {
            const formData = new FormData(document.getElementById('parameters-form'));
            const parameters = {
                confidence_threshold: parseFloat(formData.get('confidence-threshold-param')) / 100,
                historical_depth: parseInt(formData.get('historical-depth')),
                algorithm_mode: formData.get('algorithm-mode'),
                max_iterations: parseInt(formData.get('max-iterations')),
                use_frequency_analysis: formData.get('use-frequency-analysis') === 'on',
                use_pattern_detection: formData.get('use-pattern-detection') === 'on',
                enable_ensemble: formData.get('enable-ensemble') === 'on'
            };
            
            const response = await fetch(`${CONFIG.API_BASE_URL}/api/config/parameters`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(parameters)
            });
            
            if (response.ok) {
                this.showToast('Parâmetros salvos com sucesso!', 'success');
                AppState.predictionParameters = parameters;
            } else {
                throw new Error('Erro ao salvar parâmetros');
            }
        } catch (error) {
            console.error('Erro ao salvar parâmetros:', error);
            this.showToast('Erro ao salvar parâmetros', 'error');
        }
    }
    
    // Resetar parâmetros para valores padrão
    resetParameters() {
        const defaultParams = {
            'confidence-threshold-param': 75,
            'historical-depth': 100,
            'algorithm-mode': 'balanced',
            'max-iterations': 1000,
            'use-frequency-analysis': true,
            'use-pattern-detection': true,
            'enable-ensemble': false
        };
        
        Object.entries(defaultParams).forEach(([key, value]) => {
            const element = document.getElementById(key);
            if (element) {
                if (element.type === 'checkbox') {
                    element.checked = value;
                } else {
                    element.value = value;
                }
                
                // Disparar evento para atualizar displays
                element.dispatchEvent(new Event('input'));
            }
        });
        
        this.showToast('Parâmetros resetados para valores padrão', 'info');
    }
    
    // Carregar parâmetros salvos
    async loadParameters() {
        try {
            const response = await fetch(`${CONFIG.API_BASE_URL}/api/config/parameters`);
            const parameters = await response.json();
            
            // Atualizar formulário com parâmetros carregados
            if (parameters.confidence_threshold !== undefined) {
                const slider = document.getElementById('confidence-threshold-param');
                if (slider) {
                    slider.value = parameters.confidence_threshold * 100;
                    slider.dispatchEvent(new Event('input'));
                }
            }
            
            if (parameters.historical_depth !== undefined) {
                const slider = document.getElementById('historical-depth');
                if (slider) {
                    slider.value = parameters.historical_depth;
                    slider.dispatchEvent(new Event('input'));
                }
            }
            
            if (parameters.algorithm_mode) {
                const select = document.getElementById('algorithm-mode');
                if (select) select.value = parameters.algorithm_mode;
            }
            
            if (parameters.max_iterations !== undefined) {
                const slider = document.getElementById('max-iterations');
                if (slider) {
                    slider.value = parameters.max_iterations;
                    slider.dispatchEvent(new Event('input'));
                }
            }
            
            // Checkboxes
            ['use-frequency-analysis', 'use-pattern-detection', 'enable-ensemble'].forEach(id => {
                const checkbox = document.getElementById(id);
                const paramKey = id.replace(/-/g, '_');
                if (checkbox && parameters[paramKey] !== undefined) {
                    checkbox.checked = parameters[paramKey];
                }
            });
            
            AppState.predictionParameters = parameters;
            this.showToast('Parâmetros carregados com sucesso!', 'success');
            
        } catch (error) {
            console.error('Erro ao carregar parâmetros:', error);
            this.showToast('Erro ao carregar parâmetros', 'error');
        }
    }
    
    // Aplicar parâmetros e gerar nova predição
    async applyParameters() {
        try {
            // Primeiro salvar os parâmetros
            await this.saveParameters();
            
            // Depois gerar nova predição com os parâmetros aplicados
            await this.generatePredictionWithParameters();
            
        } catch (error) {
            console.error('Erro ao aplicar parâmetros:', error);
            this.showToast('Erro ao aplicar parâmetros', 'error');
        }
    }
    
    // Gerar predição com parâmetros específicos
    async generatePredictionWithParameters() {
        try {
            this.showLoading(true);
            
            const selectedNumbers = Array.from(document.querySelectorAll('.number-checkbox:checked'))
                .map(cb => parseInt(cb.value));
            
            const requestData = {
                selected_numbers: selectedNumbers,
                parameters: AppState.predictionParameters || {}
            };
            
            const response = await fetch(`${CONFIG.API_BASE_URL}/api/predictions/generate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });
            
            if (response.ok) {
                const prediction = await response.json();
                this.updateMainPrediction(prediction);
                this.showToast('Nova predição gerada com parâmetros aplicados!', 'success');
                
                // Atualizar gráficos e estatísticas
                this.updateCharts();
                this.loadDetailedStats();
            } else {
                throw new Error('Erro ao gerar predição');
            }
            
        } catch (error) {
            console.error('Erro ao gerar predição com parâmetros:', error);
            this.showToast('Erro ao gerar predição', 'error');
        } finally {
            this.showLoading(false);
        }
    }

    // Carregar notificações
    async loadNotifications(page = 1, limit = 10) {
        try {
            const filters = this.getNotificationFilters();
            const queryParams = new URLSearchParams({
                page: page.toString(),
                limit: limit.toString(),
                ...filters
            });
            
            const response = await fetch(`${CONFIG.API_BASE_URL}/api/notifications?${queryParams}`);
            const data = await response.json();
            
            this.renderNotifications(data.notifications);
            this.updateNotificationPagination(data.pagination);
            this.updateNotificationCount(data.unread_count);
            
        } catch (error) {
            console.error('Erro ao carregar notificações:', error);
            this.showToast('Erro ao carregar notificações', 'error');
        }
    }
    
    // Obter filtros de notificação
    getNotificationFilters() {
        const filters = {};
        
        const typeFilter = document.getElementById('notification-type-filter');
        if (typeFilter && typeFilter.value !== 'all') {
            filters.type = typeFilter.value;
        }
        
        const statusFilter = document.getElementById('notification-status-filter');
        if (statusFilter && statusFilter.value !== 'all') {
            filters.status = statusFilter.value;
        }
        
        const periodFilter = document.getElementById('notification-period-filter');
        if (periodFilter && periodFilter.value !== 'all') {
            filters.period = periodFilter.value;
        }
        
        return filters;
    }
    
    // Renderizar lista de notificações
    renderNotifications(notifications) {
        const container = document.getElementById('notifications-list');
        if (!container) return;
        
        if (notifications.length === 0) {
            container.innerHTML = `
                <div class="notification-item">
                    <p class="notification-message">Nenhuma notificação encontrada.</p>
                </div>
            `;
            return;
        }
        
        container.innerHTML = notifications.map(notification => `
            <div class="notification-item ${notification.read ? '' : 'unread'}" 
                 data-id="${notification.id}" 
                 onclick="lotofacilApp.markNotificationAsRead('${notification.id}')">
                <div class="notification-header">
                    <h4 class="notification-title">${notification.title}</h4>
                    <span class="notification-time">${this.formatRelativeTime(notification.created_at)}</span>
                </div>
                <p class="notification-message">${notification.message}</p>
                <span class="notification-type ${notification.type}">${notification.type}</span>
            </div>
        `).join('');
    }
    
    // Atualizar paginação de notificações
    updateNotificationPagination(pagination) {
        const container = document.querySelector('.notifications-pagination');
        if (!container) return;
        
        const { current_page, total_pages, total_items } = pagination;
        
        // Atualizar informações
        const info = container.querySelector('.pagination-info');
        if (info) {
            info.textContent = `Página ${current_page} de ${total_pages} (${total_items} total)`;
        }
        
        // Atualizar botões
        const prevBtn = container.querySelector('[data-page="prev"]');
        const nextBtn = container.querySelector('[data-page="next"]');
        
        if (prevBtn) {
            prevBtn.disabled = current_page <= 1;
            prevBtn.dataset.page = (current_page - 1).toString();
        }
        
        if (nextBtn) {
            nextBtn.disabled = current_page >= total_pages;
            nextBtn.dataset.page = (current_page + 1).toString();
        }
        
        // Atualizar botões numerados
        const numberBtns = container.querySelectorAll('.pagination-btn:not([data-page="prev"]):not([data-page="next"])');
        numberBtns.forEach(btn => {
            btn.classList.remove('active');
            if (parseInt(btn.dataset.page) === current_page) {
                btn.classList.add('active');
            }
        });
    }
    
    // Atualizar contador de notificações
    updateNotificationCount(count) {
        const badge = document.getElementById('notification-count');
        if (badge) {
            badge.textContent = count > 0 ? count.toString() : '';
            badge.style.display = count > 0 ? 'inline-block' : 'none';
        }
    }
    
    // Marcar notificação como lida
    async markNotificationAsRead(notificationId) {
        try {
            await fetch(`${CONFIG.API_BASE_URL}/api/notifications/${notificationId}/read`, {
                method: 'POST'
            });
            
            // Atualizar UI
            const notificationElement = document.querySelector(`[data-id="${notificationId}"]`);
            if (notificationElement) {
                notificationElement.classList.remove('unread');
            }
            
            // Atualizar contador
            const currentCount = parseInt(document.getElementById('notification-count').textContent) || 0;
            this.updateNotificationCount(Math.max(0, currentCount - 1));
            
        } catch (error) {
            console.error('Erro ao marcar notificação como lida:', error);
        }
    }
    
    // Filtrar notificações
    filterNotifications() {
        this.loadNotifications(1); // Resetar para primeira página
    }
    
    // Salvar configurações de notificação
    async saveNotificationSettings() {
        try {
            const settings = {};
            
            // Coletar configurações dos toggles
            document.querySelectorAll('.toggle-switch').forEach(toggle => {
                const settingName = toggle.closest('.setting-item').dataset.setting;
                if (settingName) {
                    settings[settingName] = toggle.classList.contains('active');
                }
            });
            
            await fetch(`${CONFIG.API_BASE_URL}/api/notifications/settings`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(settings)
            });
            
            this.showToast('Configurações de notificação salvas!', 'success');
            
        } catch (error) {
            console.error('Erro ao salvar configurações:', error);
            this.showToast('Erro ao salvar configurações', 'error');
        }
    }
    
    // Carregar configurações de notificação
    async loadNotificationSettings() {
        try {
            const response = await fetch(`${CONFIG.API_BASE_URL}/api/notifications/settings`);
            const settings = await response.json();
            
            // Aplicar configurações aos toggles
            Object.entries(settings).forEach(([key, value]) => {
                const toggle = document.querySelector(`[data-setting="${key}"] .toggle-switch`);
                if (toggle) {
                    toggle.classList.toggle('active', value);
                }
            });
            
        } catch (error) {
            console.error('Erro ao carregar configurações de notificação:', error);
        }
    }
    
    // Manipular nova notificação via WebSocket
    handleNewNotification(notification) {
        // Atualizar contador
        const currentCount = parseInt(document.getElementById('notification-count').textContent) || 0;
        this.updateNotificationCount(currentCount + 1);
        
        // Mostrar toast se a seção de notificações não estiver ativa
        if (AppState.currentSection !== 'notifications') {
            this.showToast(notification.title, notification.type);
        } else {
            // Recarregar notificações se estiver na seção
            this.loadNotifications();
        }
        
        // Reproduzir som de notificação se habilitado
        const soundEnabled = document.querySelector('[data-setting="sound_notifications"] .toggle-switch');
        if (soundEnabled && soundEnabled.classList.contains('active')) {
            this.playNotificationSound();
        }
    }
    
    // Reproduzir som de notificação
    playNotificationSound() {
        try {
            const audio = new Audio('data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBSuBzvLZiTYIG2m98OScTgwOUarm7blmGgU7k9n1unEiBC13yO/eizEIHWq+8+OWT');
            audio.volume = 0.3;
            audio.play().catch(() => {}); // Ignorar erros de reprodução
        } catch (error) {
            // Ignorar erros de áudio
        }
    }

    // Controle do menu mobile
    toggleMobileMenu() {
        const sidebar = document.getElementById('sidebar');
        const overlay = document.getElementById('mobile-overlay');
        
        if (sidebar && overlay) {
            sidebar.classList.toggle('mobile-open');
            overlay.classList.toggle('active');
            document.body.classList.toggle('mobile-menu-open');
        }
    }
    
    closeMobileMenu() {
        const sidebar = document.getElementById('sidebar');
        const overlay = document.getElementById('mobile-overlay');
        
        if (sidebar && overlay) {
            sidebar.classList.remove('mobile-open');
            overlay.classList.remove('active');
            document.body.classList.remove('mobile-menu-open');
        }
    }

    // Cleanup ao sair da página
    destroy() {
        AppState.updateIntervals.forEach(interval => clearInterval(interval));
        if (AppState.socket) {
            AppState.socket.disconnect();
        }
        Object.values(AppState.charts).forEach(chart => {
            if (chart && typeof chart.destroy === 'function') {
                chart.destroy();
            }
        });
    }
}

// Inicializar aplicação quando o DOM estiver carregado
document.addEventListener('DOMContentLoaded', () => {
    window.lotofacilApp = new LotofacilApp();
});

// Cleanup ao sair da página
window.addEventListener('beforeunload', () => {
    if (window.lotofacilApp) {
        window.lotofacilApp.destroy();
    }
});

// Adicionar estilos CSS dinâmicos para status badges
const style = document.createElement('style');
style.textContent = `
    .status-badge {
        padding: 0.25rem 0.5rem;
        border-radius: 0.375rem;
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
    }
    
    .status-badge.success {
        background-color: var(--success-color);
        color: white;
    }
    
    .status-badge.warning {
        background-color: var(--warning-color);
        color: white;
    }
    
    .status-badge.error {
        background-color: var(--error-color);
        color: white;
    }
    
    .metric-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid var(--border-color);
    }
    
    .metric-item:last-child {
        border-bottom: none;
    }
    
    .metric-label {
        color: var(--text-secondary);
    }
    
    .metric-value {
        font-weight: 600;
        color: var(--text-primary);
    }
    
    .toast-content {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .notification-item.unread {
        background-color: var(--bg-tertiary);
        border-left: 3px solid var(--primary-color);
    }
`;
document.head.appendChild(style);