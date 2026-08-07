(function () {
  "use strict";

  const data = window.BENCHMARK_DATA;
  if (!data) return;

  const featured = data.providers
    .filter((provider) => provider.role === "featured")
    .sort((left, right) => right.macro - left.macro);
  const colors = {
    ink: "#161a1d",
    muted: "#5d615f",
    grid: "#d7d3c9",
    card: "#fbfaf6",
    correct: "#0d776e",
    partial: "#78a8bd",
    blank: "#c9c6bd",
    wrong: "#b9362c",
    dynamic: "#2d6ea3",
    static: "#ae5a91",
    conditional: "#d68a2f"
  };
  const providerColors = {
    honcho: "#0d776e",
    mem0: "#2d6ea3",
    supermemory: "#d68a2f",
    retaindb: "#7a614a",
    hindsight: "#74558c",
    "hindsight-all": "#74558c",
    openviking: "#6c7773",
    mnemosyne: "#b9362c"
  };

  const percent = (value, digits = 1) => `${(value * 100).toFixed(digits)}%`;
  const number = (value, digits = 0) => Number(value).toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits
  });
  const escapeHtml = (value) => String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");

  function validateData() {
    const errors = [];
    data.providers.forEach((provider) => {
      const outcomeTotal = Object.values(provider.outcomes).reduce((sum, value) => sum + value, 0);
      if (outcomeTotal !== data.meta.questionsPerWave) errors.push(`${provider.name}: outcomes total ${outcomeTotal}`);
      if (provider.role === "featured") {
        const sessionTotal = data.sessionSeries[provider.id].reduce((sum, row) => sum + row[0], 0);
        if (sessionTotal !== data.meta.questionsPerWave) errors.push(`${provider.name}: session total ${sessionTotal}`);
        ["dynamic", "static", "conditional"].forEach((type) => {
          const row = data.conflictOutcomes[provider.id][type];
          const classified = row.correct + row.partial + row.blank + row.incorrect;
          if (classified !== row.N) errors.push(`${provider.name} ${type}: outcomes total ${classified}`);
        });
      }
    });
    document.documentElement.dataset.validation = errors.length ? "failed" : "passed";
    if (errors.length) console.error("Benchmark report data failed validation", errors);
  }

  function chartCanvas(rootId, height, ariaLabel) {
    const root = document.getElementById(rootId);
    if (!root) return null;
    root.innerHTML = `<div class="chart-canvas-wrap" style="height:${height}px"><canvas role="img" aria-label="${escapeHtml(ariaLabel)}"></canvas></div>`;
    return root.querySelector("canvas");
  }

  function commonOptions() {
    return {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : 280 },
      interaction: { mode: "nearest", intersect: false },
      plugins: {
        legend: {
          position: "bottom",
          labels: {
            color: colors.muted,
            usePointStyle: true,
            boxWidth: 9,
            boxHeight: 9,
            padding: 18,
            font: { family: "ui-monospace, SFMono-Regular, Consolas, monospace", size: 11, weight: 700 }
          }
        },
        tooltip: {
          backgroundColor: colors.ink,
          titleColor: "#fff",
          bodyColor: "#fff",
          padding: 11,
          cornerRadius: 3,
          titleFont: { family: "system-ui", size: 12, weight: 700 },
          bodyFont: { family: "ui-monospace, SFMono-Regular, Consolas, monospace", size: 11 }
        }
      },
      scales: {
        x: {
          grid: { color: colors.grid, borderDash: [2, 5] },
          border: { color: "#8d908c" },
          ticks: { color: colors.muted, font: { family: "ui-monospace, SFMono-Regular, Consolas, monospace", size: 10 } },
          title: { color: colors.muted, display: true, font: { family: "ui-monospace, SFMono-Regular, Consolas, monospace", size: 11, weight: 700 } }
        },
        y: {
          grid: { color: colors.grid, borderDash: [2, 5] },
          border: { color: "#8d908c" },
          ticks: { color: colors.muted, font: { family: "ui-monospace, SFMono-Regular, Consolas, monospace", size: 10 } },
          title: { color: colors.muted, display: true, font: { family: "ui-monospace, SFMono-Regular, Consolas, monospace", size: 11, weight: 700 } }
        }
      }
    };
  }

  function pointLabelPlugin(id) {
    return {
      id,
      afterDatasetsDraw(chart) {
        const { ctx, chartArea } = chart;
        ctx.save();
        ctx.font = "700 10px system-ui, sans-serif";
        ctx.textBaseline = "middle";
        ctx.lineJoin = "round";
        chart.data.datasets.forEach((dataset, datasetIndex) => {
          if (!chart.isDatasetVisible(datasetIndex)) return;
          const point = chart.getDatasetMeta(datasetIndex).data[0];
          if (!point) return;
          const { x, y } = point.getProps(["x", "y"], true);
          const label = dataset.label;
          const width = ctx.measureText(label).width;
          const placeLeft = x + width + 13 > chartArea.right;
          const tx = placeLeft ? Math.max(chartArea.left, x - width - 9) : Math.min(chartArea.right - width, x + 9);
          const labelOffsets = { Honcho: -10, mem0: 0, Supermemory: -11, Hindsight: 11, RetainDB: 0, OpenViking: -11, Mnemosyne: 11 };
          const offset = labelOffsets[label] ?? (datasetIndex % 2 ? -8 : 8);
          const ty = Math.max(chartArea.top + 8, Math.min(chartArea.bottom - 8, y + offset));
          ctx.strokeStyle = "rgba(251, 250, 246, 0.96)";
          ctx.lineWidth = 4;
          ctx.strokeText(label, tx, ty);
          ctx.fillStyle = colors.ink;
          ctx.fillText(label, tx, ty);
        });
        ctx.restore();
      }
    };
  }

  function renderOutcomeChart() {
    const canvas = chartCanvas("outcome-chart", 410, "Stacked answer outcomes for seven featured memory providers. Use the legend to filter outcome categories.");
    if (!canvas || !window.Chart) return;
    const total = data.meta.questionsPerWave;
    new Chart(canvas, {
      type: "bar",
      data: {
        labels: featured.map((provider) => provider.name),
        datasets: [
          { label: "Correct", data: featured.map((p) => p.outcomes.correct / total * 100), backgroundColor: colors.correct },
          { label: "Partial", data: featured.map((p) => p.outcomes.partial / total * 100), backgroundColor: colors.partial },
          { label: "Blank / uncertain", data: featured.map((p) => p.outcomes.blank / total * 100), backgroundColor: colors.blank },
          { label: "Wrong", data: featured.map((p) => p.outcomes.incorrect / total * 100), backgroundColor: colors.wrong }
        ]
      },
      options: {
        ...commonOptions(),
        indexAxis: "y",
        scales: {
          x: { ...commonOptions().scales.x, stacked: true, min: 0, max: 100, title: { ...commonOptions().scales.x.title, text: "Share of questions" }, ticks: { ...commonOptions().scales.x.ticks, callback: (value) => `${value}%` } },
          y: { ...commonOptions().scales.y, stacked: true, grid: { display: false }, ticks: { ...commonOptions().scales.y.ticks, font: { family: "system-ui", size: 11, weight: 700 } } }
        },
        plugins: {
          ...commonOptions().plugins,
          tooltip: { ...commonOptions().plugins.tooltip, callbacks: { label: (context) => `${context.dataset.label}: ${context.raw.toFixed(1)}%` } }
        }
      }
    });
  }

  function renderConflictOutcomeChart() {
    const canvas = chartCanvas("conflict-outcome-chart", 430, "Answer outcomes by provider for the selected conflict type. Use the selector to change conflict type and the legend to filter outcomes.");
    const select = document.getElementById("conflict-outcome-select");
    const description = document.getElementById("conflict-outcome-description");
    if (!canvas || !select || !window.Chart) return;

    const labels = {
      dynamic: "Dynamic questions: the correct answer must preserve the order of a real update.",
      static: "Static questions: a later false statement must not replace the stable fact.",
      conditional: "Conditional questions: the value must remain attached to the situation in which it applies."
    };
    const values = (type, key) => featured.map((provider) => {
      const row = data.conflictOutcomes[provider.id][type];
      return row[key] / row.N * 100;
    });
    const options = commonOptions();
    const chart = new Chart(canvas, {
      type: "bar",
      data: {
        labels: featured.map((provider) => provider.name),
        datasets: [
          { label: "Correct", data: values(select.value, "correct"), backgroundColor: colors.correct },
          { label: "Partial", data: values(select.value, "partial"), backgroundColor: colors.partial },
          { label: "Blank / uncertain", data: values(select.value, "blank"), backgroundColor: colors.blank },
          { label: "Wrong", data: values(select.value, "incorrect"), backgroundColor: colors.wrong }
        ]
      },
      options: {
        ...options,
        indexAxis: "y",
        scales: {
          x: { ...options.scales.x, stacked: true, min: 0, max: 100, title: { ...options.scales.x.title, text: "Share of questions in the selected conflict type" }, ticks: { ...options.scales.x.ticks, callback: (value) => `${value}%` } },
          y: { ...options.scales.y, stacked: true, grid: { display: false }, ticks: { ...options.scales.y.ticks, font: { family: "system-ui", size: 11, weight: 700 } } }
        },
        plugins: {
          ...options.plugins,
          tooltip: { ...options.plugins.tooltip, callbacks: { label: (context) => `${context.dataset.label}: ${context.raw.toFixed(1)}%` } }
        }
      }
    });

    const update = () => {
      const type = select.value;
      ["correct", "partial", "blank", "incorrect"].forEach((key, index) => {
        chart.data.datasets[index].data = values(type, key);
      });
      description.textContent = labels[type];
      chart.update();
    };
    select.addEventListener("change", update);
    update();
  }

  function renderScatter(rootId, config) {
    const canvas = chartCanvas(rootId, 380, `${config.ariaLabel} Use the legend to hide or show providers.`);
    if (!canvas || !window.Chart) return;
    const options = commonOptions();
    new Chart(canvas, {
      type: "scatter",
      plugins: [pointLabelPlugin(`${rootId}-point-labels`)],
      data: {
        datasets: featured.map((provider) => ({
          label: provider.short,
          data: [{ x: config.x(provider), y: provider.macro }],
          backgroundColor: providerColors[provider.id],
          borderColor: colors.card,
          borderWidth: 2,
          pointRadius: 6,
          pointHoverRadius: 8
        }))
      },
      options: {
        ...options,
        scales: {
          x: { ...options.scales.x, min: config.xMin, max: config.xMax, title: { ...options.scales.x.title, text: config.xTitle }, ticks: { ...options.scales.x.ticks, callback: config.xTick } },
          y: { ...options.scales.y, min: 0.1, max: 0.52, title: { ...options.scales.y.title, text: "Macro answer score (−1 to +1)" } }
        },
        plugins: {
          ...options.plugins,
          legend: { ...options.plugins.legend, labels: { ...options.plugins.legend.labels, padding: 10 } },
          tooltip: {
            ...options.plugins.tooltip,
            callbacks: {
              label: (context) => `${context.dataset.label}: ${config.tooltipX(context.raw.x)}, macro score ${context.raw.y.toFixed(3)}`
            }
          }
        }
      }
    });
  }

  function renderConflictChart() {
    const canvas = chartCanvas("conflict-chart", 430, "Grouped answer scores by conflict type for seven featured providers. Use the legend to filter conflict types.");
    if (!canvas || !window.Chart) return;
    const options = commonOptions();
    new Chart(canvas, {
      type: "bar",
      data: {
        labels: featured.map((provider) => provider.name),
        datasets: [
          { label: "Dynamic", data: featured.map((provider) => provider.dynamic), backgroundColor: colors.dynamic },
          { label: "Static", data: featured.map((provider) => provider.static), backgroundColor: colors.static },
          { label: "Conditional", data: featured.map((provider) => provider.conditional), backgroundColor: colors.conditional }
        ]
      },
      options: {
        ...options,
        indexAxis: "y",
        scales: {
          x: { ...options.scales.x, min: -0.25, max: 0.85, title: { ...options.scales.x.title, text: "Penalty answer score (−1 to +1)" } },
          y: { ...options.scales.y, grid: { display: false }, ticks: { ...options.scales.y.ticks, font: { family: "system-ui", size: 11, weight: 700 } } }
        },
        plugins: {
          ...options.plugins,
          tooltip: { ...options.plugins.tooltip, callbacks: { label: (context) => `${context.dataset.label}: ${Number(context.raw).toFixed(3)}` } }
        }
      }
    });
  }

  function renderSessionCharts() {
    const root = document.getElementById("session-charts");
    if (!root) return;
    root.innerHTML = featured.map((provider) => {
      const series = data.sessionSeries[provider.id];
      return `<article class="session-card">
        <h3>${escapeHtml(provider.name)}</h3>
        <p>${percent(series[1][1] / series[1][0])} correct / ${percent(series[1][4] / series[1][0])} wrong in sessions 6–10 → ${percent(series[9][1] / series[9][0])} / ${percent(series[9][4] / series[9][0])} in sessions 46–50</p>
        <div class="session-canvas-wrap"><canvas id="session-${provider.id}" role="img" aria-label="${escapeHtml(provider.name)} correct and wrong answer percentages across five-session bins. Use the legend to filter correct or wrong."></canvas></div>
      </article>`;
    }).join("");

    if (!window.Chart) return;
    featured.forEach((provider) => {
      const series = data.sessionSeries[provider.id];
      const options = commonOptions();
      new Chart(document.getElementById(`session-${provider.id}`), {
        type: "line",
        data: {
          labels: data.sessionBins,
          datasets: [
            { label: "Correct", data: series.map((row) => row[1] / row[0] * 100), borderColor: colors.correct, backgroundColor: colors.correct, pointRadius: 2.5, pointHoverRadius: 5, borderWidth: 2.5, tension: 0.18 },
            { label: "Wrong", data: series.map((row) => row[4] / row[0] * 100), borderColor: colors.wrong, backgroundColor: colors.wrong, pointRadius: 2.5, pointHoverRadius: 5, borderWidth: 2.25, borderDash: [6, 4], tension: 0.18 }
          ]
        },
        options: {
          ...options,
          scales: {
            x: { ...options.scales.x, grid: { display: false }, title: { ...options.scales.x.title, text: "Session bin" }, ticks: { ...options.scales.x.ticks, maxRotation: 0, callback: function (value, index) { return [0, 5, 10].includes(index) ? this.getLabelForValue(value) : ""; } } },
            y: { ...options.scales.y, min: 0, max: 80, title: { ...options.scales.y.title, text: "Share of questions (%)" }, ticks: { ...options.scales.y.ticks, callback: (value) => `${value}%` } }
          },
          plugins: {
            ...options.plugins,
            legend: { ...options.plugins.legend, labels: { ...options.plugins.legend.labels, padding: 10 } },
            tooltip: { ...options.plugins.tooltip, callbacks: { label: (context) => `${context.dataset.label}: ${context.raw.toFixed(1)}%` } }
          }
        }
      });
    });
  }

  function renderProviderCards() {
    const root = document.getElementById("provider-cards");
    if (!root) return;
    root.innerHTML = featured.map((provider) => `<article class="provider-card">
      <div class="provider-card-header">
        <div>
          <h3>${escapeHtml(provider.name)}</h3>
          <div class="provider-score">${provider.macro.toFixed(3)} macro score · ${percent(provider.outcomes.incorrect / data.meta.questionsPerWave)} wrong</div>
        </div>
        <span class="wordmark-mark" style="background:${providerColors[provider.id]}" aria-hidden="true">${escapeHtml(provider.short.slice(0, 2).toUpperCase())}</span>
      </div>
      <p class="provider-verdict"><span class="provider-field">Result</span>${escapeHtml(provider.verdict)}</p>
      <p class="provider-process"><span class="provider-field">Tested setup</span>${escapeHtml(provider.tested)}</p>
      <p class="provider-caveat"><span class="provider-field">Benchmark caveat</span>${escapeHtml(provider.caveat)}</p>
      <details class="provider-history">
        <summary>Integration and timeout notes</summary>
        <dl class="provider-note-list">
          <div><dt>Integration work</dt><dd>${escapeHtml(provider.process)}</dd></div>
          <div><dt>Timeout and failure behavior</dt><dd>${escapeHtml(provider.timeout)}</dd></div>
        </dl>
      </details>
      <a class="provider-link" href="${escapeHtml(provider.repo)}">Public upstream repository →</a>
    </article>`).join("");
  }

  function renderCostChart() {
    const canvas = chartCanvas("cost-chart", 460, "Model-token workload per dialogue turn for seven featured providers, separated into cached input, uncached input, and output. The horizontal scale is logarithmic; use the legend to filter token classes.");
    if (!canvas || !window.Chart) return;
    const outputPerTurn = featured.map((provider) => provider.tokens.outputTotal / data.meta.dialogueTurns);
    const options = commonOptions();
    new Chart(canvas, {
      type: "bar",
      data: {
        labels: featured.map((provider) => provider.name),
        datasets: [
          { label: "Cached input", data: featured.map((provider) => provider.tokens.cached || null), backgroundColor: colors.partial },
          { label: "Uncached input", data: featured.map((provider) => provider.tokens.uncached), backgroundColor: colors.dynamic },
          { label: "Output", data: outputPerTurn, backgroundColor: colors.conditional }
        ]
      },
      options: {
        ...options,
        indexAxis: "y",
        scales: {
          x: {
            ...options.scales.x,
            type: "logarithmic",
            min: 10,
            max: 10000,
            title: { ...options.scales.x.title, text: "Model tokens per dialogue turn (log scale)" },
            ticks: {
              ...options.scales.x.ticks,
              callback: (value) => [10, 100, 1000, 10000].includes(Number(value)) ? number(value) : ""
            },
            afterBuildTicks: (axis) => {
              axis.ticks = axis.ticks.filter((tick) => [10, 100, 1000, 10000].includes(Number(tick.value)));
            }
          },
          y: { ...options.scales.y, grid: { display: false }, ticks: { ...options.scales.y.ticks, font: { family: "system-ui", size: 11, weight: 700 } } }
        },
        plugins: {
          ...options.plugins,
          tooltip: {
            ...options.plugins.tooltip,
            callbacks: {
              label: (context) => `${context.dataset.label}: ${number(context.raw, 1)} / turn`,
              afterBody: (items) => featured[items[0].dataIndex].tokens.caveat
            }
          }
        }
      }
    });

    const notes = document.getElementById("cost-caveats");
    if (notes) {
      notes.innerHTML = featured.map((provider) => `<li><strong>${escapeHtml(provider.name)}:</strong> ${escapeHtml(provider.tokens.caveat)}</li>`).join("");
    }
  }

  function renderCostScoreChart() {
    const canvas = chartCanvas("cost-score-chart", 440, "Scatter plot of macro answer score against weighted model-token workload per dialogue turn. Lower workload and higher score is preferable.");
    if (!canvas || !window.Chart) return;
    const providers = featured;
    const weighted = (provider) => (
      0.1 * provider.tokens.cached
      + provider.tokens.uncached
      + 5 * (provider.tokens.outputTotal / data.meta.dialogueTurns)
    );
    const options = commonOptions();
    new Chart(canvas, {
      type: "scatter",
      plugins: [pointLabelPlugin("cost-score-point-labels")],
      data: {
        datasets: providers.map((provider) => ({
          label: provider.short,
          data: [{ x: weighted(provider), y: provider.macro }],
          backgroundColor: providerColors[provider.id],
          borderColor: colors.card,
          borderWidth: 2,
          pointStyle: "circle",
          pointRadius: 6,
          pointHoverRadius: 9
        }))
      },
      options: {
        ...options,
        scales: {
          x: {
            ...options.scales.x,
            min: 0,
            max: 12000,
            title: { ...options.scales.x.title, text: "Weighted workload units per dialogue turn (0.1 : 1 : 5)" },
            ticks: { ...options.scales.x.ticks, callback: (value) => number(value) }
          },
          y: { ...options.scales.y, min: 0.1, max: 0.52, title: { ...options.scales.y.title, text: "Macro answer score (−1 to +1)" } }
        },
        plugins: {
          ...options.plugins,
          legend: { ...options.plugins.legend, labels: { ...options.plugins.legend.labels, padding: 10 } },
          tooltip: {
            ...options.plugins.tooltip,
            callbacks: {
              label: (context) => `${context.dataset.label}: macro score ${context.raw.y.toFixed(3)}, weighted workload ${number(context.raw.x, 0)} / turn`
            }
          }
        }
      }
    });
  }

  function renderOutcomeTable() {
    const table = document.getElementById("outcome-table");
    if (!table) return;
    const providers = featured;
    table.innerHTML = `<caption>Featured provider outcomes and headline metrics.</caption>
      <thead><tr><th scope="col">Provider</th><th scope="col">Macro score</th><th scope="col">Correct</th><th scope="col">Partial</th><th scope="col">Blank</th><th scope="col">Wrong</th></tr></thead>
      <tbody>${providers.map((provider) => `<tr>
        <th scope="row">${escapeHtml(provider.name)}</th>
        <td>${provider.macro.toFixed(3)}</td>
        <td>${percent(provider.outcomes.correct / data.meta.questionsPerWave)}</td><td>${percent(provider.outcomes.partial / data.meta.questionsPerWave)}</td>
        <td>${percent(provider.outcomes.blank / data.meta.questionsPerWave)}</td><td>${percent(provider.outcomes.incorrect / data.meta.questionsPerWave)}</td>
      </tr>`).join("")}</tbody>`;
  }

  function renderSessionTable() {
    const table = document.getElementById("session-table");
    if (!table) return;
    table.innerHTML = `<caption>Correct and wrong answer percentages by five-session bin.</caption>
      <thead><tr><th scope="col">Provider</th>${data.sessionBins.map((bin) => `<th scope="col">${bin}</th>`).join("")}</tr></thead>
      <tbody>${featured.flatMap((provider) => {
        const series = data.sessionSeries[provider.id];
        return [
          `<tr><th scope="row">${escapeHtml(provider.name)} — correct</th>${series.map((row) => `<td>${percent(row[1] / row[0])}</td>`).join("")}</tr>`,
          `<tr><th scope="row">${escapeHtml(provider.name)} — wrong</th>${series.map((row) => `<td>${percent(row[4] / row[0])}</td>`).join("")}</tr>`
        ];
      }).join("")}</tbody>`;
  }

  function renderConflictOutcomeTable() {
    const table = document.getElementById("conflict-outcome-table");
    if (!table) return;
    const labels = { dynamic: "Dynamic", static: "Static", conditional: "Conditional" };
    table.innerHTML = `<caption>Penalty-rubric answer outcomes within each conflict type.</caption>
      <thead><tr><th scope="col">Provider</th><th scope="col">Type</th><th scope="col">Correct</th><th scope="col">Partial</th><th scope="col">Blank</th><th scope="col">Wrong</th></tr></thead>
      <tbody>${featured.flatMap((provider) => ["dynamic", "static", "conditional"].map((type) => {
        const row = data.conflictOutcomes[provider.id][type];
        return `<tr><th scope="row">${escapeHtml(provider.name)}</th><td>${labels[type]}</td><td>${percent(row.correct / row.N)}</td><td>${percent(row.partial / row.N)}</td><td>${percent(row.blank / row.N)}</td><td>${percent(row.incorrect / row.N)}</td></tr>`;
      })).join("")}</tbody>`;
  }

  function renderFallback() {
    if (window.Chart) return;
    document.querySelectorAll(".svg-chart, .outcome-chart, .conflict-chart, .cost-chart").forEach((root) => {
      root.innerHTML = "<p class=\"chart-error\">The chart library did not load. Exact values are available in the data tables.</p>";
    });
  }

  function setupMobileNav() {
    document.querySelectorAll(".mobile-nav a").forEach((link) => {
      link.addEventListener("click", () => link.closest("details")?.removeAttribute("open"));
    });
  }

  validateData();
  setupMobileNav();
  renderOutcomeChart();
  renderConflictOutcomeChart();
  renderScatter("risk-scatter", {
    ariaLabel: "Scatter plot of macro answer score against wrong-answer rate for seven providers.",
    x: (provider) => provider.outcomes.incorrect / data.meta.questionsPerWave * 100,
    xMin: 0,
    xMax: 22,
    xTitle: "Overall wrong answers (% of 3,750)",
    xTick: (value) => `${value}%`,
    tooltipX: (value) => `${value.toFixed(1)}% wrong`
  });
  renderConflictChart();
  renderSessionCharts();
  renderProviderCards();
  renderCostChart();
  renderCostScoreChart();
  renderOutcomeTable();
  renderSessionTable();
  renderConflictOutcomeTable();
  renderFallback();
})();
