// builds the group results page from the pipeline's summary. plain functions that return
// text, so they can be tested without a browser. everything that came from a file is
// escaped before it goes into html: connection names are whatever the user typed.

export const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
));

const known = (value) => value !== null && value !== undefined && !Number.isNaN(value);

// three significant figures, written out in full
export function num(value, digits = 3) {
  if (!known(value)) return "";
  if (Number.isInteger(value)) return String(value);
  if (Math.abs(value) >= 10 ** digits) return value.toFixed(0);
  return String(Number(value.toPrecision(digits)));
}

export function pvalue(value) {
  if (!known(value)) return "";
  return value < 0.001 ? "<0.001" : value.toFixed(3);
}

const NAMES = {
  lesion_volume_ml: "Lesion volume (mL)",
  lesion_volume_norm_ml: "Lesion volume, adjusted for head size (mL)",
  lesion_count: "Number of lesions",
  head_scale: "Head-size factor",
  etiv_ml: "Estimated intracranial volume (mL)",
  nbss_total: "NBSS total",
  nbss_total_post: "NBSS total after treatment",
  diary_voids_per_day: "Voids per day",
  diary_urgency_per_day: "Urgency episodes per day",
  diary_incontinence_per_day: "Incontinence episodes per day",
};

// a readable name for a column of the subject table
export function label(variable) {
  if (NAMES[variable]) return NAMES[variable];
  const tract = /^tract_(.+?)_(norm_ml|ml)$/.exec(variable);
  if (tract) {
    const name = tract[1].replace(/_/g, " ");
    return tract[2] === "ml" ? `${name} lesion volume (mL)` : `${name} lesion volume, adjusted (mL)`;
  }
  return variable.replace(/_/g, " ");
}

export const QUESTIONS = {
  q1_total_burden_fc: "Total lesion burden and functional connectivity",
  q2_tract_load_fc: "Tract lesion load and functional connectivity",
  q3_detrusor_overactivity: "Lesion burden by detrusor overactivity",
  q4_symptom_severity: "Lesion burden and symptom severity",
  q5_treatment_response: "Lesion burden by treatment response",
};
const GROUPS = {
  q3_detrusor_overactivity: ["DO present", "DO absent"],
  q5_treatment_response: ["responder", "non-responder"],
};

function table(head, rows, numeric = []) {
  const cls = (i) => (numeric.includes(i) ? ' class="n"' : "");
  const th = head.map((h, i) => `<th scope="col"${cls(i)}>${h}</th>`).join("");
  const body = rows.map((row) => `<tr>${row.map((cell, i) => `<td${cls(i)}>${cell}</td>`).join("")}</tr>`).join("");
  return `<div class="tablewrap"><table><thead><tr>${th}</tr></thead><tbody>${body}</tbody></table></div>`;
}

const interval = (r) => `${num(r.estimate)} <span class="ci">(${num(r.ci_low)} to ${num(r.ci_high)})</span>`;

// ---- every scan's measurements ------------------------------------------------------

export function tractNames(measurements) {
  const names = new Set();
  measurements.forEach((row) => Object.keys(row.tracts ?? {}).forEach((name) => names.add(name)));
  return [...names];
}

export function scanRows(measurements) {
  const tracts = tractNames(measurements);
  const head = ["Scan", "Lesion volume (mL)", "Adjusted for head size (mL)", "Lesions",
    ...tracts.flatMap((t) => [`${t} (mL)`, `${t}, adjusted (mL)`]), "Head-size factor"];
  const rows = measurements.map((m) => [
    m.id, m.lesion_volume_ml, m.lesion_volume_norm_ml, m.lesion_count,
    ...tracts.flatMap((t) => [m.tracts?.[t]?.ml, m.tracts?.[t]?.norm_ml]), m.head_scale,
  ]);
  return { head, rows };
}

// tab-separated, so it pastes straight into a spreadsheet or an email
export function scansAsText(measurements) {
  const { head, rows } = scanRows(measurements);
  const line = (cells) => cells.map((c) => (known(c) ? String(c) : "")).join("\t");
  return [line(head), ...rows.map(line)].join("\n");
}

function scanTable(measurements) {
  const { head, rows } = scanRows(measurements);
  const body = rows.map(([id, ...values]) => [
    `<button type="button" class="open" data-scan="${esc(id)}">${esc(id)}</button>`,
    ...values.map((v) => num(v)),
  ]);
  return table(head.map(esc), body, head.map((_, i) => i).slice(1));
}

// ---- the group summary ---------------------------------------------------------------

function describeTable(descriptives, group, variables) {
  const rows = [];
  for (const variable of variables) {
    const d = descriptives.find((x) => x.group === group && x.variable === variable);
    if (!d) continue;
    rows.push(d.suppressed
      ? [esc(label(variable)), num(d.n), "withheld, group too small", "", ""]
      : [esc(label(variable)), num(d.n), `${num(d.mean)} ± ${num(d.sd)}`,
        `${num(d.median)} <span class="ci">(${num(d.q1)} to ${num(d.q3)})</span>`, num(d.n_nonzero)]);
  }
  return rows.length
    ? table(["Measure", "n", "Mean ± SD", "Median (IQR)", "n above zero"], rows, [1, 2, 3, 4])
    : "";
}

function fcSection(tests, alpha) {
  const families = new Map();
  for (const r of tests) {
    const key = `${r.exposure}|${r.state}`;
    if (!families.has(key)) families.set(key, []);
    families.get(key).push(r);
  }
  const overview = [...families.values()].map((rows) => {
    const ok = rows.filter((r) => r.status === "ok");
    const qs = ok.map((r) => r.q).filter(known);
    return [esc(rows[0].exposure), esc(rows[0].state), num(ok.length),
      num(ok.filter((r) => known(r.q) && r.q < alpha).length), qs.length ? pvalue(Math.min(...qs)) : ""];
  });
  let html = table(["Exposure", "Bladder state", "Connections tested", `q below ${alpha}`, "Smallest q"],
    overview, [2, 3, 4]);
  const hits = tests.filter((r) => known(r.q) && r.q < alpha).sort((a, b) => a.q - b.q);
  if (hits.length) {
    html += `<p class="note">Connections with q below ${alpha}:</p>`;
    html += table(["Exposure", "Bladder state", "Connection", "n", "rho (95% CI)", "p", "q"],
      hits.map((r) => [esc(r.exposure), esc(r.state), esc(r.connection), num(r.n), interval(r),
        pvalue(r.p), pvalue(r.q)]), [3, 4, 5, 6]);
  } else {
    html += `<p class="note">No connection reached q below ${alpha}.</p>`;
  }
  return html;
}

function correlationSection(tests) {
  return table(["Exposure", "Outcome", "Role", "n", "rho (95% CI)", "p, unadjusted"],
    tests.map((r) => [esc(r.exposure), esc(label(r.outcome)), esc(r.role), num(r.n),
      r.status === "ok" ? interval(r) : esc(r.status), pvalue(r.p)]), [3, 4, 5]);
}

function groupSection(tests, question, descriptives) {
  const [one, zero] = GROUPS[question];
  let html = table(["Outcome", `n, ${esc(one)} / ${esc(zero)}`, "Rank-biserial r (95% CI)",
    "Typical difference, mL (95% CI)", "p, unadjusted"],
  tests.map((r) => [esc(r.outcome), `${num(r.n_group1)} / ${num(r.n_group0)}`,
    r.status === "ok" ? interval(r) : esc(r.status),
    r.status === "ok"
      ? `${num(r.shift)} <span class="ci">(${num(r.shift_ci_low)} to ${num(r.shift_ci_high)})</span>` : "",
    pvalue(r.p)]), [1, 2, 3, 4]);
  const measured = [...new Set(descriptives.filter((d) => d.group === one).map((d) => d.variable))]
    .filter((v) => /_norm_ml$|^lesion_volume_ml$|^tract_[^_]+_ml$/.test(v))
    .filter((v) => !/^tract_.+_(L|R|temporal)_/.test(v));
  for (const group of [one, zero]) {
    const block = describeTable(descriptives, group, measured);
    if (block) html += `<h4>${esc(group)}</h4>${block}`;
  }
  html += `<p class="note">A positive r or difference means the ${esc(one)} group has the larger volumes.</p>`;
  return html;
}

export function reportHtml(summary, measurements) {
  const parts = [];
  if (summary?.fake_data) {
    parts.push('<p class="banner"><b>Fake data.</b> The clinical and FC tables behind these results '
      + "were made up to test the pipeline. Nothing in the group results is a finding.</p>");
  }
  if (measurements.length) {
    parts.push(`<section><h3>Each scan</h3>
      <p class="note">Select a scan to open it. Volumes are in mL.</p>${scanTable(measurements)}</section>`);
  }
  if (!summary) {
    parts.push('<section><h3>Group results</h3><p class="note">There are no group results yet. '
      + "They appear here after <code>mslb run</code> has finished.</p></section>");
    return parts.join("");
  }
  const { run, descriptives = [], tests = [] } = summary;
  const alpha = run?.settings?.stats?.alpha ?? 0.05;
  const counts = run?.counts ?? {};
  const scale = run?.settings?.stats?.volume_scale === "raw"
    ? "raw, not adjusted for head size" : "adjusted for head size";
  parts.push(`<section><h3>Cohort</h3>
    <p class="note">${num(counts.processed)} of ${num(counts.scans)} scans processed.
    ${num(counts.with_clinical)} with clinical data, ${num(counts.with_fc)} with FC data.
    Lesion volumes in the tests are ${scale}. Every test is two-sided.</p>
    ${describeTable(descriptives, "all", [...new Set(descriptives.filter((d) => d.group === "all"
    && !d.variable.startsWith("fc__") && !/^tract_.+_(L|R|temporal)_/.test(d.variable)).map((d) => d.variable))])}
    </section>`);

  let index = 0;
  for (const [question, title] of Object.entries(QUESTIONS)) {
    index += 1;
    const block = tests.filter((r) => r.question === question);
    if (!block.length) continue;
    let body;
    if (question.endsWith("_fc")) body = fcSection(block, alpha);
    else if (question === "q4_symptom_severity") body = correlationSection(block);
    else body = groupSection(block, question, descriptives);
    parts.push(`<section><h3>Question ${index}. ${esc(title)}</h3>${body}</section>`);
  }
  return parts.join("");
}

// one scan's numbers as text, for pasting into a message
export function scanAsText(meta) {
  const s = meta.stats ?? {};
  const lines = [`Scan\t${meta.id}`];
  const add = (name, value) => { if (known(value)) lines.push(`${name}\t${value}`); };
  add("Lesion volume (mL)", s.lesion_volume_ml);
  add("Lesion volume, adjusted for head size (mL)", s.lesion_volume_norm_ml);
  add("Number of lesions", s.lesion_count);
  for (const [name, load] of Object.entries(s.tracts ?? {})) {
    add(`${name} lesion volume (mL)`, load.ml);
    add(`${name} lesion volume, adjusted (mL)`, load.norm_ml);
    add(`${name}, share of the tract lesioned (%)`, load.pct);
  }
  add("Head-size factor", s.head_scale);
  return lines.join("\n");
}
