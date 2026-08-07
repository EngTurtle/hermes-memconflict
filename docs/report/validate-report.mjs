import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const reportDir = path.dirname(fileURLToPath(import.meta.url));
const benchmarkDir = path.dirname(reportDir);
const errors = [];

function csvRows(filePath) {
  const text = fs.readFileSync(filePath, "utf8").trim();
  const lines = text.split(/\r?\n/);
  const headers = lines.shift().split(",");
  return lines.map((line) => {
    const cells = [];
    let value = "";
    let quoted = false;
    for (let i = 0; i < line.length; i += 1) {
      const char = line[i];
      if (char === '"') {
        if (quoted && line[i + 1] === '"') {
          value += '"';
          i += 1;
        } else {
          quoted = !quoted;
        }
      } else if (char === "," && !quoted) {
        cells.push(value);
        value = "";
      } else {
        value += char;
      }
    }
    cells.push(value);
    return Object.fromEntries(headers.map((header, index) => [header, cells[index]]));
  });
}

const context = { window: {} };
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(reportDir, "assets", "data.js"), "utf8"), context);
const report = context.window.BENCHMARK_DATA;

const summary = csvRows(path.join(benchmarkDir, "Scores", "v5_featured_results_summary.csv"));
for (const provider of report.providers) {
  const row = summary.find((item) => item.provider === (provider.sourceName || provider.name) && item.tag === provider.tag);
  if (!row) {
    errors.push(`Missing source summary for: ${provider.name} ${provider.tag}`);
    continue;
  }
  const expected = {
    correct: Number(row.correct),
    partial: Number(row.partial_correct),
    blank: Number(row.blank),
    incorrect: Number(row.incorrect)
  };
  for (const [key, value] of Object.entries(expected)) {
    if (provider.outcomes[key] !== value) errors.push(`${provider.name} ${key}: ${provider.outcomes[key]} != ${value}`);
  }
}

const bins = csvRows(path.join(benchmarkDir, "Scores", "v5_featured_results_by_session_bin.csv"));
const featured = report.providers.filter((provider) => provider.role === "featured");
for (const provider of featured) {
  const sourceRows = bins.filter((row) => row.provider === (provider.sourceName || provider.name) && row.tag === provider.tag);
  const embeddedRows = report.sessionSeries[provider.id];
  if (sourceRows.length !== embeddedRows.length) {
    errors.push(`${provider.name}: ${embeddedRows.length} embedded bins != ${sourceRows.length} source bins`);
    continue;
  }
  sourceRows.forEach((row, index) => {
    const expected = [row.number_of_questions, row.correct, row.partial_correct, row.blank, row.incorrect].map(Number);
    if (expected.some((value, cell) => value !== embeddedRows[index][cell])) {
      errors.push(`${provider.name} bin ${index + 1}: ${embeddedRows[index].join("/")} != ${expected.join("/")}`);
    }
  });
  for (const type of ["dynamic", "static", "conditional"]) {
    const outcomes = report.conflictOutcomes[provider.id][type];
    const classified = outcomes.correct + outcomes.partial + outcomes.blank + outcomes.incorrect;
    if (classified !== outcomes.N) errors.push(`${provider.name} ${type}: ${classified} classified != N ${outcomes.N}`);
    if (outcomes.N !== report.meta.conflictCounts[type]) errors.push(`${provider.name} ${type}: N ${outcomes.N} != benchmark ${report.meta.conflictCounts[type]}`);
  }
}

const html = fs.readFileSync(path.join(reportDir, "index.html"), "utf8");
for (const required of ["outcome-chart", "conflict-outcome-chart", "conflict-outcome-select", "risk-scatter", "conflict-chart", "session-charts", "provider-cards", "cost-chart", "cost-score-chart", "cost-caveats", "outcome-table", "conflict-outcome-table", "session-table"]) {
  if (!html.includes(`id="${required}"`)) errors.push(`Missing report element #${required}`);
}
if (/github\.com\/EngTurtle|private_testing|private-testing/i.test(html)) {
  errors.push("Private repository identifier leaked into report HTML");
}
const ids = new Set([...html.matchAll(/\sid="([^"]+)"/g)].map((match) => match[1]));
for (const match of html.matchAll(/href="#([^"]+)"/g)) {
  if (!ids.has(match[1])) errors.push(`Broken internal link: #${match[1]}`);
}
for (const match of html.matchAll(/(?:src|href)="(assets\/[^"]+)"/g)) {
  const cleanAssetPath = match[1].split(/[?#]/, 1)[0];
  if (!fs.existsSync(path.join(reportDir, cleanAssetPath))) errors.push(`Missing local asset: ${cleanAssetPath}`);
}

const bannedProse = [
  "in today's rapidly evolving landscape",
  "at its core",
  "let's dive into",
  "it's worth noting that",
  "in conclusion",
  "ultimately",
  "the score is only half",
  "the store grew. so did",
  "three jobs, three rankings",
  "not a made-up dollar bill"
];
for (const phrase of bannedProse) {
  if (html.toLowerCase().includes(phrase)) errors.push(`Anti-slop phrase found: ${phrase}`);
}

const chartBundle = fs.readFileSync(path.join(reportDir, "assets", "chart.umd.min.js"), "utf8");
if (!chartBundle.slice(0, 500).includes("Chart.js v4.5.1")) errors.push("Unexpected Chart.js bundle version");

if (errors.length) {
  throw new Error(errors.join("\n"));
}

console.log(`Report validation passed: ${featured.length} selected provider waves checked against ${summary.length} source waves and ${bins.length} session-bin rows.`);
