const ThemeManager = {
    init() {
        const savedTheme = localStorage.getItem('structura-theme');
        const themeToApply = savedTheme === 'light' ? 'light' : 'dark';
        this.setTheme(themeToApply);
        
        document.getElementById('themeToggle').addEventListener('click', () => {
            const currentTheme = document.documentElement.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            this.setTheme(newTheme);
        });
    },
    setTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('structura-theme', theme);
    }
};

const SAMPLE_ABSTRACTS = [
    {
        topic: "Orthopedics",
        text: "Chronic low back pain remains a major cause of disability worldwide, with limited long-term efficacy from standard physical therapies. We aimed to evaluate whether an 8-week mindfulness-based stress reduction (MBSR) program reduces pain intensity compared to usual care. A randomized, parallel-group trial was conducted with 240 adults aged 18 to 65 years diagnosed with persistent non-specific low back pain. Participants were randomly assigned 1:1 to either the MBSR group or a waitlist control receiving standard medical care. The primary outcome was the change in back pain intensity measured on a 10-point visual analog scale at 24 weeks. A total of 218 participants completed the final 24-week follow-up assessment. The MBSR group demonstrated a significant reduction in pain scores compared to the control group (mean difference -1.4 points; 95% CI, -1.9 to -0.9; P < 0.001). No serious adverse events associated with the intervention were reported during the trial period. An 8-week MBSR program provides clinically meaningful, sustained improvement in pain and functional limitations for patients with chronic low back pain."
    },
    {
        topic: "Endocrinology",
        text: "The optimal macronutrient distribution for managing metabolic markers in recently diagnosed type 2 diabetes patients is still heavily debated. This study sought to compare the effects of a low-carbohydrate Mediterranean diet versus a standard low-fat diet on glycemic control. We designed a single-center, open-label, randomized controlled trial involving 180 treatment-naive adults. Over a 12-month period, the intervention group followed a low-carbohydrate Mediterranean diet, while the comparator group adhered to a low-fat diet. Hemoglobin A1c (HbA1c) levels and lipid profiles were evaluated at baseline, 6 months, and 12 months. At the 12-month mark, the Mediterranean diet group achieved a significantly greater reduction in HbA1c than the low-fat group (-1.2% vs -0.7%, P = 0.02). Additionally, high-density lipoprotein cholesterol levels increased more prominently in the low-carbohydrate cohort. Adherence to a low-carbohydrate Mediterranean diet leads to superior glycemic control and better lipid profiles than a conventional low-fat diet."
    },
    {
        topic: "Pulmonology",
        text: "Pediatric asthma management heavily relies on consistent medication adherence, which is traditionally poor among school-aged children. We investigated whether an interactive mobile health application could improve inhaler adherence and reduce emergency department visits. A multi-center randomized controlled trial enrolled 350 children aged 7 to 12 with moderate-to-severe persistent asthma. Families were randomized to either use the gamified smartphone app linked to an electronic inhaler sensor or receive standard asthma education. Adherence data were automatically logged via electronic sensors, and emergency visits were tracked over 6 months. The intervention group achieved an average daily inhaler adherence rate of 78%, compared to 54% in the control group (P < 0.001). Emergency department visits dropped by 45% in the app group relative to the standard care group over the 6-month tracking window. Integrating mobile health applications with electronic trackers significantly boosts pediatric asthma adherence and lowers acute care utilization."
    },
    {
        topic: "Robotics",
        text: "Autonomous vehicle navigation relies heavily on real-time object detection, but current deep learning models often struggle under heavy rain and foggy conditions. This paper introduces an adaptive feature-fusion network designed to maintain high object-detection accuracy across diverse adverse weather scenarios. We trained and evaluated our convolutional neural network architecture on a newly curated synthetic dataset containing 50,000 degraded driving images. The proposed model dynamically adjusts convolutional filter weights based on estimated ambient lighting and visibility metrics extracted from the input frame. Performance was benchmarked against three industry-standard architectures using mean Average Precision (mAP) calculations. The adaptive network achieved an mAP of 84.3% in heavy rain simulations, outperforming the baseline model by 11.5 percentage points. Processing speeds remained stable at 45 frames per second on standard edge-computing hardware. Dynamic feature-fusion networks offer a reliable and computationally efficient solution for enhancing autonomous vehicle safety in unpredictable environments."
    }
];

const ApiService = {
    getApiBaseUrl() {
        const { hostname, protocol } = window.location;

        if (hostname === 'localhost' || hostname === '127.0.0.1') {
            return 'http://localhost:8080';
        }

        const forwardedBackendHost = hostname.replace(/-\d+(?=\.app\.github\.dev$)/, '-8080');
        return `${protocol}//${forwardedBackendHost}`;
    },

    async analyzeAbstract(text) {
        const response = await fetch(`${this.getApiBaseUrl()}/api/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });

        if (!response.ok) {
            throw new Error(`Analysis request failed with status ${response.status}`);
        }

        return response.json();
    }
};

const App = {
    elements: {
        input: document.getElementById('abstract-input'),
        btnAnalyze: document.getElementById('btn-analyze'),
        btnClear: document.getElementById('btn-clear'),
        loadingState: document.getElementById('loading-state'),
        resultsContainer: document.getElementById('results-container'),
        samplesGrid: document.getElementById('samples-grid')
    },

    init() {
        ThemeManager.init();
        this.renderSamples();
        this.bindEvents();
    },

    bindEvents() {
        this.elements.btnAnalyze.addEventListener('click', () => this.handleAnalyze());
        this.elements.btnClear.addEventListener('click', () => this.handleClear());
        
        this.elements.input.addEventListener('input', () => {
            this.elements.btnAnalyze.disabled = this.elements.input.value.trim().length === 0;
        });
        this.elements.btnAnalyze.disabled = true;

        document.addEventListener('click', (e) => {
            if (e.target.closest('.raw-toggle-btn')) {
                const popover = document.querySelector('.raw-popover');
                popover.classList.toggle('active');
            } else if (!e.target.closest('.raw-output-wrapper')) {
                const popover = document.querySelector('.raw-popover.active');
                if (popover) popover.classList.remove('active');
            }
            
            const altToggle = e.target.closest('.alt-toggle');
            if (altToggle) {
                const expanded = altToggle.getAttribute('aria-expanded') === 'true';
                altToggle.setAttribute('aria-expanded', !expanded);
            }
        });
    },

    renderSamples() {
        this.elements.samplesGrid.innerHTML = SAMPLE_ABSTRACTS.map((sample, index) => `
            <div class="sample-card">
                <div class="sample-topic">${sample.topic}</div>
                <div class="sample-preview">${sample.text}</div>
                <button class="sample-btn" data-index="${index}">Try this</button>
            </div>
        `).join('');

        document.querySelectorAll('.sample-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const index = e.target.getAttribute('data-index');
                this.elements.input.value = SAMPLE_ABSTRACTS[index].text;
                this.elements.btnAnalyze.disabled = false;
                
                window.scrollTo({ top: 0, behavior: 'smooth' });
                this.elements.input.focus();
            });
        });
    },

    handleClear() {
        this.elements.input.value = '';
        this.elements.btnAnalyze.disabled = true;
        this.elements.resultsContainer.innerHTML = '';
        this.elements.input.focus();
    },

    async handleAnalyze() {
        const text = this.elements.input.value.trim();
        if (!text) return;

        this.elements.btnAnalyze.disabled = true;
        this.elements.input.disabled = true;
        this.elements.resultsContainer.innerHTML = '';
        this.elements.loadingState.classList.remove('hidden');

        try {
            const result = await ApiService.analyzeAbstract(text);
            this.renderResults(result.data, result.rawOutput);
        } catch (error) {
            console.error("Analysis failed:", error);
            alert("An error occurred during analysis. Please try again.");
        } finally {
            this.elements.loadingState.classList.add('hidden');
            this.elements.btnAnalyze.disabled = false;
            this.elements.input.disabled = false;
        }
    },

    // Generates grouped paragraph HTML where each unique category forms a block
    generateStructuredParagraphsHTML(predictions, useAlternative = false) {
        const groups = [];
        let currentGroup = null;

        predictions.forEach(item => {
            const pClass = useAlternative ? item.alternative.predictedClass : item.predictedClass;
            const pConf = useAlternative ? item.alternative.confidence : item.confidence;

            if (!currentGroup || currentGroup.category !== pClass) {
                if (currentGroup) groups.push(currentGroup);
                currentGroup = {
                    category: pClass,
                    lines: [],
                    totalConfidence: 0
                };
            }
            currentGroup.lines.push(item.text);
            currentGroup.totalConfidence += pConf;
        });
        if (currentGroup) groups.push(currentGroup);

        let html = ``;
        groups.forEach(group => {
            const meanConf = (group.totalConfidence / group.lines.length * 100).toFixed(2);
            const proseText = group.lines.join(' ');
            
            html += `
                <div class="structured-prose-block">
                    <div class="category-header">
                        <span class="category-label">${group.category}</span>
                        <div class="confidence-tooltip">Mean confidence: ${meanConf}%</div>
                    </div>
                    <span class="group-text">${proseText}</span>
                </div>
            `;
        });
        return html;
    },

    generateRawScoresTableHTML(predictions) {
        const classNames = ['BACKGROUND', 'CONCLUSIONS', 'METHODS', 'OBJECTIVE', 'RESULTS'];
        const headerHTML = classNames.map(className => `<th scope="col">${className}</th>`).join('');
        const rowsHTML = predictions.map((item, index) => {
            const scores = Array.isArray(item.rawScores) ? item.rawScores : [];
            const scoreCells = classNames.map((_, scoreIndex) => {
                const score = Number(scores[scoreIndex]);
                return `<td>${Number.isFinite(score) ? score.toFixed(6) : '-'}</td>`;
            }).join('');

            return `
                <tr>
                    <th scope="row">${index + 1}</th>
                    <td><span class="raw-label">${this.escapeHTML(item.predictedClass)}</span></td>
                    <td class="raw-line">${this.escapeHTML(item.text)}</td>
                    ${scoreCells}
                </tr>
            `;
        }).join('');

        return `
            <div class="raw-table-scroll" role="region" aria-label="Raw prediction scores" tabindex="0">
                <table class="raw-scores-table">
                    <caption class="visually-hidden">Raw probability scores for each analyzed line</caption>
                    <thead>
                        <tr>
                            <th scope="col">#</th>
                            <th scope="col">Assigned label</th>
                            <th scope="col">Line</th>
                            ${headerHTML}
                        </tr>
                    </thead>
                    <tbody>${rowsHTML}</tbody>
                </table>
            </div>
        `;
    },

    renderResults(predictions) {
        const primaryHTML = this.generateStructuredParagraphsHTML(predictions, false);
        const altHTML = this.generateStructuredParagraphsHTML(predictions, true);
        const rawScoresTableHTML = this.generateRawScoresTableHTML(predictions);

        const html = `
            <section class="results-section">
                <h2 class="section-heading">Structured abstract</h2>
                <div class="result-card glass-panel">
                    <div class="raw-output-wrapper">
                        <button class="raw-toggle-btn" aria-expanded="false" aria-controls="raw-output-popover">Raw output</button>
                        <div class="raw-popover" id="raw-output-popover">
                            ${rawScoresTableHTML}
                        </div>
                    </div>
                    
                    ${primaryHTML}
                </div> 

                <div class="alt-predictions glass-panel">
                    <button class="alt-toggle" aria-expanded="false">
                        See alternative predictions
                        <svg class="chevron" viewBox="0 0 24 24"><path d="M7 10l5 5 5-5z"/></svg>
                    </button>
                    <div class="alt-content-wrapper">
                        <div class="alt-content">
                            <div class="alt-content-inner">
                                ${altHTML}
                            </div>
                        </div>
                    </div>
                </div>
            </section>
        `;

        this.elements.resultsContainer.innerHTML = html;
    },

    escapeHTML(str) {
        return str.replace(/[&<>'"]/g, 
            tag => ({
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                "'": '&#39;',
                '"': '&quot;'
            }[tag])
        );
    }
};

document.addEventListener('DOMContentLoaded', () => {
    App.init();
});