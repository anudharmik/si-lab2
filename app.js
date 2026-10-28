/**
 * FreshTrack Food Waste Intelligence — Application Logic
 * Replicates the analytics and machine learning pipeline from the Jupyter notebook.
 */

const FreshTrackApp = (() => {
  // Application State
  const state = {
    data: null,
    model: null,
    filteredCountries: [],
    filters: {
      region: 'ALL',
      confidence: 'ALL',
      search: ''
    },
    rankingMetric: 'combined_kg',
    table: {
      search: '',
      region: 'ALL',
      sortKey: 'household_tonnes',
      sortDir: 'desc',
      page: 1,
      pageSize: 15
    },
    theme: localStorage.getItem('freshtrack-theme') || 'light',
    charts: {}
  };

  // Helper number formatters
  const fmt = {
    tonnes: (val) => {
      if (val >= 1e6) return (val / 1e6).toFixed(1) + 'M';
      if (val >= 1e3) return (val / 1e3).toFixed(1) + 'k';
      return Math.round(val).toLocaleString();
    },
    exact: (val) => Math.round(val).toLocaleString(),
    dec: (val, d = 1) => Number(val).toFixed(d),
    pct: (val) => Number(val).toFixed(1) + '%'
  };

  // Initialize App
  async function init() {
    initTheme();

    // Load data from global bundle or fetch
    if (window.FRESHTRACK_DATA && window.FRESHTRACK_MODEL) {
      state.data = window.FRESHTRACK_DATA;
      state.model = window.FRESHTRACK_MODEL;
    } else {
      try {
        const [dataRes, modelRes] = await Promise.all([
          fetch('data/dashboard_data.json'),
          fetch('data/model_trees.json')
        ]);
        state.data = await dataRes.json();
        state.model = await modelRes.json();
      } catch (err) {
        console.error('Failed loading data JSON:', err);
        return;
      }
    }

    state.filteredCountries = [...state.data.countries];

    populateFilterDropdowns();
    setupEventListeners();
    updateDashboardMetrics();
    initCharts();
    renderCountryCards();
    renderRegionalCards();
    renderTable();
    initPredictor();
  }

  // Theme Management
  function initTheme() {
    document.documentElement.setAttribute('data-theme', state.theme);
    updateThemeIcon();

    const themeToggleBtn = document.getElementById('theme-toggle');
    if (themeToggleBtn) {
      themeToggleBtn.addEventListener('click', toggleTheme);
    }
  }

  function toggleTheme() {
    state.theme = state.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', state.theme);
    localStorage.setItem('freshtrack-theme', state.theme);
    updateThemeIcon();
    updateChartTheme();
  }

  function updateThemeIcon() {
    const icon = document.getElementById('theme-icon');
    if (!icon) return;
    if (state.theme === 'light') {
      // Moon icon for switching to dark
      icon.innerHTML = `<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>`;
    } else {
      // Sun icon for switching to light
      icon.innerHTML = `
        <circle cx="12" cy="12" r="5"/>
        <line x1="12" y1="1" x2="12" y2="3"/>
        <line x1="12" y1="21" x2="12" y2="23"/>
        <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
        <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
        <line x1="1" y1="12" x2="3" y2="12"/>
        <line x1="21" y1="12" x2="23" y2="12"/>
        <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
        <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
      `;
    }
  }

  // Populate Dropdown Menus
  function populateFilterDropdowns() {
    const regionSelect = document.getElementById('filter-region');
    const tableRegionSelect = document.getElementById('table-region-filter');
    const mlRegionSelect = document.getElementById('ml-input-region');

    const regions = state.data.regions.map(r => r.region).sort();

    regions.forEach(region => {
      const opt1 = new Option(region, region);
      const opt2 = new Option(region, region);
      const opt3 = new Option(region, region);
      if (regionSelect) regionSelect.appendChild(opt1);
      if (tableRegionSelect) tableRegionSelect.appendChild(opt2);
      if (mlRegionSelect) mlRegionSelect.appendChild(opt3);
    });

    if (mlRegionSelect) {
      mlRegionSelect.value = 'Australia and New Zealand';
    }
  }

  // Event Listeners
  function setupEventListeners() {
    // Global filter listeners
    const regionFilter = document.getElementById('filter-region');
    const confFilter = document.getElementById('filter-confidence');
    const searchFilter = document.getElementById('filter-search-quick');

    if (regionFilter) {
      regionFilter.addEventListener('change', (e) => {
        state.filters.region = e.target.value;
        applyFilters();
      });
    }

    if (confFilter) {
      confFilter.addEventListener('change', (e) => {
        state.filters.confidence = e.target.value;
        applyFilters();
      });
    }

    if (searchFilter) {
      searchFilter.addEventListener('input', (e) => {
        state.filters.search = e.target.value.trim().toLowerCase();
        applyFilters();
      });
    }

    // Country browser search
    const countrySearch = document.getElementById('country-search-input');
    if (countrySearch) {
      countrySearch.addEventListener('input', (e) => {
        renderCountryCards(e.target.value.trim().toLowerCase());
      });
    }

    // Ranking metric toggle buttons
    const rankingToggles = document.querySelectorAll('#ranking-metric-toggle .tab-toggle');
    rankingToggles.forEach(btn => {
      btn.addEventListener('click', (e) => {
        rankingToggles.forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        state.rankingMetric = e.target.dataset.metric;
        updateRankingsChart();
      });
    });

    // Table search & filter
    const tableSearch = document.getElementById('table-search-input');
    const tableRegion = document.getElementById('table-region-filter');

    if (tableSearch) {
      tableSearch.addEventListener('input', (e) => {
        state.table.search = e.target.value.trim().toLowerCase();
        state.table.page = 1;
        renderTable();
      });
    }

    if (tableRegion) {
      tableRegion.addEventListener('change', (e) => {
        state.table.region = e.target.value;
        state.table.page = 1;
        renderTable();
      });
    }

    // Table header sort
    const tableHeaders = document.querySelectorAll('#dataset-table th[data-sort]');
    tableHeaders.forEach(th => {
      th.addEventListener('click', () => {
        const key = th.dataset.sort;
        if (state.table.sortKey === key) {
          state.table.sortDir = state.table.sortDir === 'asc' ? 'desc' : 'asc';
        } else {
          state.table.sortKey = key;
          state.table.sortDir = (key === 'name' || key === 'region' || key === 'confidence') ? 'asc' : 'desc';
        }
        renderTable();
      });
    });

    // Navigation scrollspy
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('section[id]');
    window.addEventListener('scroll', () => {
      let current = '';
      sections.forEach(sec => {
        const top = sec.offsetTop - 120;
        if (window.scrollY >= top) {
          current = sec.getAttribute('id');
        }
      });
      navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === '#' + current) {
          link.classList.add('active');
        }
      });
    });
  }

  // Filter application
  function applyFilters() {
    const { region, confidence, search } = state.filters;

    state.filteredCountries = state.data.countries.filter(c => {
      const matchRegion = (region === 'ALL') || (c.region === region);
      const matchConf = (confidence === 'ALL') || (c.confidence === confidence);
      const matchSearch = (!search) || c.name.toLowerCase().includes(search);
      return matchRegion && matchConf && matchSearch;
    });

    // Toggle reset button
    const resetBtn = document.getElementById('btn-reset-filters');
    const counterTag = document.getElementById('filter-counter-tag');
    const isFiltered = (region !== 'ALL' || confidence !== 'ALL' || search !== '');

    if (resetBtn) resetBtn.style.display = isFiltered ? 'inline-flex' : 'none';
    if (counterTag) {
      counterTag.textContent = `${state.filteredCountries.length} of ${state.data.countries.length} countries active`;
    }

    updateDashboardMetrics();
    updateRankingsChart();
    renderCountryCards();
  }

  function resetFilters() {
    state.filters = { region: 'ALL', confidence: 'ALL', search: '' };
    document.getElementById('filter-region').value = 'ALL';
    document.getElementById('filter-confidence').value = 'ALL';
    document.getElementById('filter-search-quick').value = '';
    applyFilters();
  }

  // Dashboard Metrics Recalculation
  function updateDashboardMetrics() {
    const list = state.filteredCountries;
    if (!list.length) {
      document.getElementById('kpi-total-tonnes').textContent = '0';
      document.getElementById('kpi-household-share').textContent = '0%';
      document.getElementById('kpi-avg-combined').textContent = '0';
      document.getElementById('kpi-countries-count').textContent = '0';
      return;
    }

    const totalTonnes = list.reduce((acc, c) => acc + c.total_tonnes, 0);
    const hhTonnes = list.reduce((acc, c) => acc + c.household_tonnes, 0);
    const avgCombined = list.reduce((acc, c) => acc + c.combined_kg, 0) / list.length;
    const hhShare = totalTonnes > 0 ? (hhTonnes / totalTonnes) * 100 : 0;

    document.getElementById('kpi-total-tonnes').textContent = fmt.tonnes(totalTonnes);
    document.getElementById('kpi-household-share').textContent = fmt.pct(hhShare);
    document.getElementById('kpi-household-tonnes').textContent = `(${fmt.tonnes(hhTonnes)} t)`;
    document.getElementById('kpi-avg-combined').textContent = fmt.dec(avgCombined);
    document.getElementById('kpi-countries-count').textContent = list.length;

    // Meter widths
    const globalTotal = state.data.metrics.global_total_tonnes;
    const meterPct = Math.min(100, Math.round((totalTonnes / globalTotal) * 100));
    const totalMeter = document.getElementById('kpi-total-meter');
    if (totalMeter) totalMeter.style.width = meterPct + '%';

    const hhMeter = document.getElementById('kpi-household-meter');
    if (hhMeter) hhMeter.style.width = Math.min(100, hhShare) + '%';

    const coverageBadge = document.getElementById('kpi-coverage-badge');
    if (coverageBadge) {
      coverageBadge.textContent = `${Math.round((list.length / 214) * 100)}% of UNEP`;
    }
  }

  // Charts
  function initCharts() {
    Chart.defaults.font.family = "'Plus Jakarta Sans', sans-serif";
    Chart.defaults.font.weight = '600';
    Chart.defaults.color = getChartTextColor();
    Chart.defaults.borderColor = getChartGridColor();

    initRankingsChart();
    initSourcesDonutChart();
    initSourcesBarsChart();
    initRegionalBarsChart();
    initActualVsPredictedChart();
  }

  function getChartTextColor() {
    return state.theme === 'dark' ? '#f2f7f4' : '#0a130e';
  }

  function getChartGridColor() {
    return state.theme === 'dark' ? 'rgba(255, 255, 255, 0.08)' : 'rgba(28, 44, 34, 0.09)';
  }

  function getChartTooltipOptions(customCallbacks = {}) {
    const isDark = state.theme === 'dark';
    return {
      backgroundColor: isDark ? 'rgba(15, 22, 17, 0.96)' : 'rgba(255, 255, 255, 0.98)',
      titleColor: isDark ? '#ffffff' : '#0a130e',
      bodyColor: isDark ? '#e0ebe3' : '#14221a',
      borderColor: isDark ? 'rgba(62, 130, 99, 0.35)' : 'rgba(28, 44, 34, 0.18)',
      borderWidth: 1,
      padding: 12,
      cornerRadius: 8,
      titleFont: { family: "'Outfit', sans-serif", size: 13, weight: '800' },
      bodyFont: { family: "'Plus Jakarta Sans', sans-serif", size: 12, weight: '600' },
      callbacks: customCallbacks
    };
  }

  function updateChartTheme() {
    const textColor = getChartTextColor();
    const gridColor = getChartGridColor();

    // Reset global Chart.js defaults so all charts pick up new theme colors
    Chart.defaults.color = textColor;
    Chart.defaults.borderColor = gridColor;
    const isDark = state.theme === 'dark';

    if (state.charts.sourcesDonut && state.charts.sourcesDonut.data.datasets[0]) {
      state.charts.sourcesDonut.data.datasets[0].borderColor = isDark ? '#151d18' : '#ffffff';
    }
    if (state.charts.actualVsPred && state.charts.actualVsPred.data.datasets[1]) {
      state.charts.actualVsPred.data.datasets[1].borderColor = isDark ? 'rgba(255, 255, 255, 0.25)' : 'rgba(28, 44, 34, 0.22)';
    }

    Object.values(state.charts).forEach(chart => {
      if (!chart) return;
      if (chart.options.scales) {
        if (chart.options.scales.x) {
          chart.options.scales.x.ticks.color = textColor;
          chart.options.scales.x.grid.color = gridColor;
          if (chart.options.scales.x.title) chart.options.scales.x.title.color = textColor;
        }
        if (chart.options.scales.y) {
          chart.options.scales.y.ticks.color = textColor;
          chart.options.scales.y.grid.color = gridColor;
          if (chart.options.scales.y.title) chart.options.scales.y.title.color = textColor;
        }
      }
      if (chart.options.plugins) {
        if (chart.options.plugins.legend && chart.options.plugins.legend.labels) {
          chart.options.plugins.legend.labels.color = textColor;
        }
        if (chart.options.plugins.tooltip) {
          chart.options.plugins.tooltip.backgroundColor = isDark ? 'rgba(21, 29, 24, 0.95)' : 'rgba(255, 255, 255, 0.97)';
          chart.options.plugins.tooltip.titleColor = isDark ? '#f2f7f4' : '#17241d';
          chart.options.plugins.tooltip.bodyColor = isDark ? '#9cb0a4' : '#526359';
          chart.options.plugins.tooltip.borderColor = isDark ? 'rgba(62, 130, 99, 0.25)' : 'rgba(28, 44, 34, 0.12)';
        }
      }
      chart.update();
    });
  }

  // Chart 1: Country Rankings
  function initRankingsChart() {
    const ctx = document.getElementById('chart-country-rankings');
    if (!ctx) return;

    state.charts.rankings = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: [],
        datasets: [{
          label: 'Waste',
          data: [],
          backgroundColor: '#255a40',
          hoverBackgroundColor: '#327553',
          borderRadius: 6
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: getChartTooltipOptions({
            label: (ctx) => {
              const metric = state.rankingMetric;
              const unit = metric.includes('tonnes') ? 'tonnes/year' : 'kg/person/year';
              return ` ${ctx.parsed.x.toLocaleString()} ${unit}`;
            }
          })
        },
        scales: {
          x: {
            grid: { color: getChartGridColor() },
            ticks: {
              color: getChartTextColor(),
              callback: (val) => {
                if (state.rankingMetric.includes('tonnes')) return fmt.tonnes(val);
                return val;
              }
            }
          },
          y: {
            grid: { display: false },
            ticks: { color: getChartTextColor(), font: { weight: '600' } }
          }
        }
      }
    });

    updateRankingsChart();
  }

  function updateRankingsChart() {
    const chart = state.charts.rankings;
    if (!chart) return;

    const metric = state.rankingMetric;
    const sorted = [...state.filteredCountries]
      .sort((a, b) => b[metric] - a[metric])
      .slice(0, 10)
      .reverse();

    chart.data.labels = sorted.map(c => c.name);
    chart.data.datasets[0].data = sorted.map(c => c[metric]);

    const titleEl = document.getElementById('rankings-chart-title');
    const subEl = document.getElementById('rankings-chart-subtitle');

    if (metric === 'combined_kg') {
      if (titleEl) titleEl.textContent = 'Top 10 Countries by Combined Food Waste';
      if (subEl) subEl.textContent = 'Ranked by combined kilograms per person per year';
      chart.data.datasets[0].backgroundColor = '#255a40';
      chart.data.datasets[0].hoverBackgroundColor = '#327553';
    } else if (metric === 'household_tonnes') {
      if (titleEl) titleEl.textContent = 'Top 10 Countries by Household Volume';
      if (subEl) subEl.textContent = 'Ranked by total annual tonnes of household waste';
      chart.data.datasets[0].backgroundColor = '#486a7d';
      chart.data.datasets[0].hoverBackgroundColor = '#5c849c';
    } else {
      if (titleEl) titleEl.textContent = 'Top 10 Countries by Household Per-Capita';
      if (subEl) subEl.textContent = 'Ranked by annual kilograms per person from private homes';
      chart.data.datasets[0].backgroundColor = '#9e7a42';
      chart.data.datasets[0].hoverBackgroundColor = '#b58f53';
    }

    chart.update();
  }

  // Chart 2: Sources Donut
  function initSourcesDonutChart() {
    const ctx = document.getElementById('chart-sources-donut');
    if (!ctx) return;

    const sources = state.data.sources;
    state.charts.sourcesDonut = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: ['Household', 'Food Service', 'Retail'],
        datasets: [{
          data: [
            sources.shares_percent.household,
            sources.shares_percent.foodservice,
            sources.shares_percent.retail
          ],
          backgroundColor: ['#255a40', '#486a7d', '#9e7a42'],
          borderColor: state.theme === 'dark' ? '#151d18' : '#ffffff',
          borderWidth: 2,
          hoverOffset: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '66%',
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              color: getChartTextColor(),
              font: { weight: '600', size: 12 },
              padding: 16
            }
          },
          tooltip: getChartTooltipOptions({
            label: (ctx) => ` ${ctx.label}: ${ctx.parsed}% of global waste`
          })
        }
      }
    });
  }

  // Chart 3: Sources Bars
  function initSourcesBarsChart() {
    const ctx = document.getElementById('chart-sources-bars');
    if (!ctx) return;

    const avg = state.data.sources.averages_kg;
    state.charts.sourcesBars = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: ['Household', 'Food Service', 'Retail'],
        datasets: [{
          label: 'Average kg / person / year',
          data: [avg.household, avg.foodservice, avg.retail],
          backgroundColor: ['#255a40', '#486a7d', '#9e7a42'],
          borderRadius: 8
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: getChartTooltipOptions({
            label: (ctx) => ` Average: ${ctx.parsed.y} kg/person/year`
          })
        },
        scales: {
          x: { grid: { display: false }, ticks: { color: getChartTextColor(), font: { weight: '600' } } },
          y: {
            grid: { color: getChartGridColor() },
            ticks: {
              color: getChartTextColor(),
              callback: (v) => v + ' kg'
            }
          }
        }
      }
    });
  }

  // Chart 4: Regional Bars
  function initRegionalBarsChart() {
    const ctx = document.getElementById('chart-regional-bars');
    if (!ctx) return;

    const regions = [...state.data.regions].reverse();
    state.charts.regionalBars = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: regions.map(r => r.region),
        datasets: [{
          label: 'Average combined food waste (kg/capita/year)',
          data: regions.map(r => r.avg_combined_kg),
          backgroundColor: regions.map(r => r.avg_combined_kg >= 140 ? '#255a40' : '#478366'),
          borderRadius: 5
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        layout: {
          padding: { left: 16, right: 12 }
        },
        plugins: {
          legend: { display: false },
          tooltip: getChartTooltipOptions({
            label: (ctx) => ` Combined: ${ctx.parsed.x} kg/person/year`
          })
        },
        scales: {
          x: {
            grid: { color: getChartGridColor() },
            ticks: { color: getChartTextColor(), callback: (v) => v + ' kg' }
          },
          y: {
            grid: { display: false },
            ticks: { color: getChartTextColor(), font: { size: 11, weight: '500' }, autoSkip: false }
          }
        }
      }
    });
  }

  // Chart 5: Actual vs Predicted
  function initActualVsPredictedChart() {
    const ctx = document.getElementById('chart-actual-vs-predicted');
    if (!ctx) return;

    const testPts = state.data.test_comparison.map(p => ({
      x: p.actual / 1e6,
      y: p.predicted / 1e6,
      country: p.country,
      region: p.region
    }));

    const maxVal = Math.max(...testPts.map(p => Math.max(p.x, p.y))) * 1.1;

    state.charts.actualVsPred = new Chart(ctx, {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: 'Test Countries',
            data: testPts,
            backgroundColor: '#255a40',
            borderColor: '#3e8263',
            borderWidth: 1.5,
            pointRadius: 6,
            pointHoverRadius: 8
          },
          {
            label: 'Perfect Match (45° Identity)',
            data: [{ x: 0, y: 0 }, { x: maxVal, y: maxVal }],
            type: 'line',
            borderColor: state.theme === 'dark' ? 'rgba(255, 255, 255, 0.25)' : 'rgba(28, 44, 34, 0.22)',
            borderDash: [5, 5],
            fill: false,
            pointRadius: 0
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            labels: { color: getChartTextColor(), font: { size: 12 } }
          },
          tooltip: getChartTooltipOptions({
            title: (items) => {
              const raw = items[0].raw;
              return raw.country ? `${raw.country} (${raw.region})` : 'Parity Line';
            },
            label: (ctx) => {
              const raw = ctx.raw;
              if (!raw.country) return '';
              return [
                ` Actual: ${fmt.dec(raw.x, 2)}M tonnes`,
                ` Predicted: ${fmt.dec(raw.y, 2)}M tonnes`
              ];
            }
          })
        },
        scales: {
          x: {
            title: { display: true, text: 'Actual Household Waste (Million tonnes/year)', color: getChartTextColor() },
            grid: { color: getChartGridColor() },
            ticks: { color: getChartTextColor() }
          },
          y: {
            title: { display: true, text: 'Predicted Household Waste (Million tonnes/year)', color: getChartTextColor() },
            grid: { color: getChartGridColor() },
            ticks: { color: getChartTextColor() }
          }
        }
      }
    });
  }

  // Country Cards List
  function renderCountryCards(searchQuery = '') {
    const container = document.getElementById('country-cards-container');
    if (!container) return;

    let list = state.filteredCountries;
    if (searchQuery) {
      list = list.filter(c => c.name.toLowerCase().includes(searchQuery));
    }

    if (!list.length) {
      container.innerHTML = `<div style="text-align:center; padding: 30px; color: var(--text-muted);">No countries match your search.</div>`;
      return;
    }

    container.innerHTML = list.slice(0, 30).map(c => `
      <div class="country-item-card" onclick="FreshTrackApp.openDrawer('${c.name.replace(/'/g, "\\'")}')">
        <div class="country-item-info">
          <span class="country-item-name">${c.name}</span>
          <span class="country-item-region">${c.region}</span>
        </div>
        <div class="country-item-stats">
          <div class="stat-group">
            <span class="stat-label">Combined</span>
            <span class="stat-val">${c.combined_kg} <span style="font-size:0.7rem; font-weight:500;">kg/p</span></span>
          </div>
          <div class="stat-group">
            <span class="stat-label">Household</span>
            <span class="stat-val">${fmt.tonnes(c.household_tonnes)}</span>
          </div>
          <div>
            ${getConfidenceBadgeHTML(c.confidence)}
          </div>
        </div>
      </div>
    `).join('');
  }

  // Regional Cards List
  function renderRegionalCards() {
    const container = document.getElementById('regional-cards-list');
    if (!container) return;

    const regions = state.data.regions.slice(0, 8);
    container.innerHTML = regions.map(r => `
      <div class="country-item-card" style="cursor: default;">
        <div class="country-item-info">
          <span class="country-item-name">${r.region}</span>
          <span class="country-item-region">${r.country_count} nations • ${fmt.tonnes(r.total_waste_tonnes)} t total</span>
        </div>
        <div class="country-item-stats">
          <div class="stat-group">
            <span class="stat-label">Avg Combined</span>
            <span class="stat-val" style="color: var(--accent-mint);">${r.avg_combined_kg} <span style="font-size:0.7rem; font-weight:500;">kg/p</span></span>
          </div>
        </div>
      </div>
    `).join('');
  }

  function getConfidenceBadgeHTML(conf) {
    if (conf === 'High Confidence') return `<span class="conf-badge high">High</span>`;
    if (conf === 'Medium Confidence') return `<span class="conf-badge medium">Medium</span>`;
    if (conf === 'Low Confidence') return `<span class="conf-badge low">Low</span>`;
    return `<span class="conf-badge very-low">Very Low</span>`;
  }

  // Country Detail Drawer
  function openDrawer(countryName) {
    const country = state.data.countries.find(c => c.name === countryName);
    if (!country) return;

    document.getElementById('drawer-country-name').textContent = country.name;
    document.getElementById('drawer-region').textContent = country.region;
    document.getElementById('drawer-comb-val').textContent = `${country.combined_kg} kg`;
    document.getElementById('drawer-hh-tonnes').textContent = `${fmt.tonnes(country.household_tonnes)} t`;
    document.getElementById('drawer-m49').textContent = country.m49 || 'N/A';

    // Confidence badge
    const confCont = document.getElementById('drawer-conf-container');
    if (confCont) confCont.innerHTML = getConfidenceBadgeHTML(country.confidence);

    // Global comparison text
    const globalAvg = state.data.metrics.avg_combined_kg;
    const diffPct = Math.round(((country.combined_kg - globalAvg) / globalAvg) * 100);
    const benchmarkEl = document.getElementById('drawer-benchmark-text');
    if (benchmarkEl) {
      if (diffPct > 0) {
        benchmarkEl.innerHTML = `<span style="color:var(--accent-amber);">+${diffPct}% above</span> the global average (${globalAvg} kg/person).`;
      } else if (diffPct < 0) {
        benchmarkEl.innerHTML = `<span style="color:var(--accent-mint);">${Math.abs(diffPct)}% below</span> the global average (${globalAvg} kg/person).`;
      } else {
        benchmarkEl.innerHTML = `Matches the global average (${globalAvg} kg/person).`;
      }
    }

    // Source sector bars
    const totalKg = country.household_kg + country.foodservice_kg + country.retail_kg;
    const hhShare = totalKg > 0 ? Math.round((country.household_kg / totalKg) * 100) : 0;
    const fsShare = totalKg > 0 ? Math.round((country.foodservice_kg / totalKg) * 100) : 0;
    const retShare = totalKg > 0 ? Math.round((country.retail_kg / totalKg) * 100) : 0;

    document.getElementById('drawer-hh-meter-label').textContent = `${country.household_kg} kg (${hhShare}%)`;
    document.getElementById('drawer-hh-bar').style.width = hhShare + '%';

    document.getElementById('drawer-fs-meter-label').textContent = `${country.foodservice_kg} kg (${fsShare}%)`;
    document.getElementById('drawer-fs-bar').style.width = fsShare + '%';

    document.getElementById('drawer-ret-meter-label').textContent = `${country.retail_kg} kg (${retShare}%)`;
    document.getElementById('drawer-ret-bar').style.width = retShare + '%';

    // Source URL
    const sourceLink = document.getElementById('drawer-source-link');
    if (sourceLink) {
      sourceLink.href = country.source || 'https://www.unep.org/resources/report/unep-food-waste-index-report-2021';
    }

    document.getElementById('country-drawer-backdrop').classList.add('active');
  }

  function closeDrawer(e) {
    if (e && e.target !== e.currentTarget && !e.target.closest('.drawer-header button')) return;
    document.getElementById('country-drawer-backdrop').classList.remove('active');
  }

  // Machine Learning Estimator Simulation
  function initPredictor() {
    const sliderComb = document.getElementById('slider-comb');
    const sliderHh = document.getElementById('slider-hh');
    const sliderFs = document.getElementById('slider-fs');
    const sliderRet = document.getElementById('slider-ret');

    const regionSelect = document.getElementById('ml-input-region');
    const confSelect = document.getElementById('ml-input-confidence');

    const updateHandler = () => {
      document.getElementById('badge-comb-val').textContent = sliderComb.value + ' kg/capita';
      document.getElementById('badge-hh-val').textContent = sliderHh.value + ' kg/capita';
      document.getElementById('badge-fs-val').textContent = sliderFs.value + ' kg/capita';
      document.getElementById('badge-ret-val').textContent = sliderRet.value + ' kg/capita';
      runPrediction();
    };

    [sliderComb, sliderHh, sliderFs, sliderRet].forEach(s => {
      if (s) s.addEventListener('input', updateHandler);
    });

    if (regionSelect) regionSelect.addEventListener('change', runPrediction);
    if (confSelect) confSelect.addEventListener('change', runPrediction);

    runPrediction();
  }

  function runPrediction() {
    if (!state.model) return;

    const region = document.getElementById('ml-input-region').value;
    const confidence = document.getElementById('ml-input-confidence').value;
    const comb = parseFloat(document.getElementById('slider-comb').value);
    const hh = parseFloat(document.getElementById('slider-hh').value);
    const fs = parseFloat(document.getElementById('slider-fs').value);
    const ret = parseFloat(document.getElementById('slider-ret').value);

    // Build feature vector matching Scikit-learn ColumnTransformer
    const x = [comb, hh, ret, fs];

    // One-hot categories from model
    const [confCats, regionCats] = state.model.categories;
    confCats.forEach(c => {
      x.push(confidence === c ? 1.0 : 0.0);
    });
    regionCats.forEach(r => {
      x.push(region === r ? 1.0 : 0.0);
    });

    // Evaluate Random Forest 200 trees
    const forest = state.model.forest;
    let sumPreds = 0;
    const numTrees = forest.length;

    for (let t = 0; t < numTrees; t++) {
      const tree = forest[t];
      let node = 0;
      while (tree.left[node] !== -1) {
        const featIdx = tree.feature[node];
        if (x[featIdx] <= tree.thresh[node]) {
          node = tree.left[node];
        } else {
          node = tree.right[node];
        }
      }
      sumPreds += tree.values[node];
    }

    const predictedVal = Math.round(sumPreds / numTrees);
    document.getElementById('ml-output-number').textContent = predictedVal.toLocaleString();
  }

  function loadPreset(presetName) {
    const pills = document.querySelectorAll('.preset-pill');
    pills.forEach(p => p.classList.remove('active'));

    const activeBtn = document.getElementById(`preset-${presetName}`);
    if (activeBtn) activeBtn.classList.add('active');

    if (presetName === 'australia') {
      // Step 10 notebook example
      document.getElementById('ml-input-region').value = 'Australia and New Zealand';
      document.getElementById('ml-input-confidence').value = 'High Confidence';
      document.getElementById('slider-comb').value = 133;
      document.getElementById('slider-hh').value = 102;
      document.getElementById('slider-fs').value = 22;
      document.getElementById('slider-ret').value = 9;
    } else if (presetName === 'global') {
      document.getElementById('ml-input-region').value = 'Western Europe';
      document.getElementById('ml-input-confidence').value = 'Medium Confidence';
      document.getElementById('slider-comb').value = 127;
      document.getElementById('slider-hh').value = 84;
      document.getElementById('slider-fs').value = 27;
      document.getElementById('slider-ret').value = 15;
    } else if (presetName === 'high') {
      document.getElementById('ml-input-region').value = 'Western Asia';
      document.getElementById('ml-input-confidence').value = 'Low Confidence';
      document.getElementById('slider-comb').value = 175;
      document.getElementById('slider-hh').value = 120;
      document.getElementById('slider-fs').value = 35;
      document.getElementById('slider-ret').value = 20;
    } else if (presetName === 'low') {
      document.getElementById('ml-input-region').value = 'Northern Europe';
      document.getElementById('ml-input-confidence').value = 'High Confidence';
      document.getElementById('slider-comb').value = 75;
      document.getElementById('slider-hh').value = 45;
      document.getElementById('slider-fs').value = 18;
      document.getElementById('slider-ret').value = 12;
    }

    document.getElementById('badge-comb-val').textContent = document.getElementById('slider-comb').value + ' kg/capita';
    document.getElementById('badge-hh-val').textContent = document.getElementById('slider-hh').value + ' kg/capita';
    document.getElementById('badge-fs-val').textContent = document.getElementById('slider-fs').value + ' kg/capita';
    document.getElementById('badge-ret-val').textContent = document.getElementById('slider-ret').value + ' kg/capita';

    runPrediction();
  }

  // Data Table & Pagination
  function renderTable() {
    const { search, region, sortKey, sortDir, page, pageSize } = state.table;

    let rows = [...state.data.countries];

    if (search) {
      rows = rows.filter(r => r.name.toLowerCase().includes(search) || r.region.toLowerCase().includes(search));
    }
    if (region !== 'ALL') {
      rows = rows.filter(r => r.region === region);
    }

    // Sort rows
    rows.sort((a, b) => {
      let valA = a[sortKey];
      let valB = b[sortKey];
      if (typeof valA === 'string') {
        return sortDir === 'asc' ? valA.localeCompare(valB) : valB.localeCompare(valA);
      }
      return sortDir === 'asc' ? valA - valB : valB - valA;
    });

    const totalRows = rows.length;
    const totalPages = Math.ceil(totalRows / pageSize) || 1;
    const curPage = Math.min(page, totalPages);
    const startIdx = (curPage - 1) * pageSize;
    const pageRows = rows.slice(startIdx, startIdx + pageSize);

    // Update header sort classes
    document.querySelectorAll('#dataset-table th[data-sort]').forEach(th => {
      th.classList.remove('sorted-asc', 'sorted-desc');
      if (th.dataset.sort === sortKey) {
        th.classList.add(sortDir === 'asc' ? 'sorted-asc' : 'sorted-desc');
      }
    });

    // Populate rows
    const tbody = document.getElementById('table-body');
    if (tbody) {
      if (!pageRows.length) {
        tbody.innerHTML = `<tr><td colspan="8" style="text-align:center; padding: 24px; color: var(--text-muted);">No records found.</td></tr>`;
      } else {
        tbody.innerHTML = pageRows.map(c => `
          <tr onclick="FreshTrackApp.openDrawer('${c.name.replace(/'/g, "\\'")}')">
            <td style="font-weight: 600;">${c.name}</td>
            <td style="color: var(--text-secondary);">${c.region}</td>
            <td class="num-cell">${c.combined_kg}</td>
            <td class="num-cell">${c.household_kg}</td>
            <td class="num-cell" style="font-weight:700; color:var(--accent-mint);">${c.household_tonnes.toLocaleString()}</td>
            <td class="num-cell">${c.retail_kg}</td>
            <td class="num-cell">${c.foodservice_kg}</td>
            <td>${getConfidenceBadgeHTML(c.confidence)}</td>
          </tr>
        `).join('');
      }
    }

    // Info text
    const infoEl = document.getElementById('table-pagination-info');
    if (infoEl) {
      const endIdx = Math.min(startIdx + pageSize, totalRows);
      infoEl.textContent = totalRows > 0
        ? `Showing ${startIdx + 1}–${endIdx} of ${totalRows} countries`
        : '0 countries';
    }

    // Pagination buttons
    const btnContainer = document.getElementById('table-pagination-buttons');
    if (btnContainer) {
      let btnsHTML = `
        <button class="btn-page" ${curPage === 1 ? 'disabled' : ''} onclick="FreshTrackApp.goToPage(${curPage - 1})">‹</button>
      `;

      for (let p = 1; p <= totalPages; p++) {
        if (p === 1 || p === totalPages || (p >= curPage - 1 && p <= curPage + 1)) {
          btnsHTML += `<button class="btn-page ${p === curPage ? 'active' : ''}" onclick="FreshTrackApp.goToPage(${p})">${p}</button>`;
        } else if (p === curPage - 2 || p === curPage + 2) {
          btnsHTML += `<span style="padding: 0 4px; color: var(--text-muted);">...</span>`;
        }
      }

      btnsHTML += `
        <button class="btn-page" ${curPage === totalPages ? 'disabled' : ''} onclick="FreshTrackApp.goToPage(${curPage + 1})">›</button>
      `;
      btnContainer.innerHTML = btnsHTML;
    }
  }

  function goToPage(p) {
    state.table.page = p;
    renderTable();
  }

  // Export functions
  function exportCSV() {
    const list = state.filteredCountries.length ? state.filteredCountries : state.data.countries;
    const headers = [
      'Country',
      'Region',
      'Combined (kg/capita/year)',
      'Household estimate (kg/capita/year)',
      'Household estimate (tonnes/year)',
      'Retail estimate (kg/capita/year)',
      'Food service estimate (kg/capita/year)',
      'Confidence in estimate',
      'M49 code'
    ];

    const rows = list.map(c => [
      `"${c.name}"`,
      `"${c.region}"`,
      c.combined_kg,
      c.household_kg,
      c.household_tonnes,
      c.retail_kg,
      c.foodservice_kg,
      `"${c.confidence}"`,
      c.m49
    ]);

    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', 'freshtrack_food_waste_data.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  function exportJSON() {
    const list = state.filteredCountries.length ? state.filteredCountries : state.data.countries;
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(list, null, 2));
    const link = document.createElement('a');
    link.setAttribute('href', dataStr);
    link.setAttribute('download', 'freshtrack_food_waste_data.json');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  // Public API
  return {
    init,
    resetFilters,
    openDrawer,
    closeDrawer,
    loadPreset,
    goToPage,
    exportCSV,
    exportJSON
  };
})();

// Auto-run on DOM ready
document.addEventListener('DOMContentLoaded', FreshTrackApp.init);
