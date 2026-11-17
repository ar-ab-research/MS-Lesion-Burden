import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import { check } from "../scripts/check-public.mjs";
import {
  esc, label, num, pvalue, reportHtml, scanAsText, scanRows, scansAsText,
} from "../src/report.mjs";

const measurements = [
  { id: "s1", lesion_volume_ml: 38.51, lesion_volume_norm_ml: 57.93, lesion_count: 14, head_scale: 1.5042,
    tracts: { ATR: { ml: 1.34, norm_ml: 2.02, pct: 12.9 }, SLF: { ml: 0.39, norm_ml: 0.58, pct: 4.4 } } },
  { id: "s2", lesion_volume_ml: 2.21, lesion_volume_norm_ml: null, lesion_count: 13, head_scale: null,
    tracts: { ATR: { ml: null, norm_ml: null, pct: null }, SLF: { ml: null, norm_ml: null, pct: null } } },
];

const test_row = (extra) => ({
  question: "q1_total_burden_fc", role: "primary", exposure: "total lesion volume", outcome: "FC full: a__b",
  state: "full", connection: "a__b", test: "spearman", n: 20, estimate: -0.6, ci_low: -0.8, ci_high: -0.2,
  p: 0.004, q: 0.02, status: "ok", ...extra,
});

const summary = {
  fake_data: false,
  markdown: "# Group summary",
  run: { versions: { mslb: "0.1.0" }, counts: { processed: 20, scans: 20, with_clinical: 20, with_fc: 19 },
    settings: { stats: { alpha: 0.05, volume_scale: "normalized" } } },
  descriptives: [
    { group: "all", variable: "lesion_volume_norm_ml", n: 20, mean: 32.2, sd: 26.9, median: 25.9, q1: 5.8, q3: 53.3, suppressed: false, n_nonzero: 20 },
    { group: "all", variable: "fc__full__a__b", n: 19, mean: 0.2, sd: 0.1, median: 0.2, q1: 0.1, q3: 0.3, suppressed: false, n_nonzero: 19 },
    { group: "DO present", variable: "lesion_volume_norm_ml", n: 2, mean: null, sd: null, median: null, q1: null, q3: null, suppressed: true, n_nonzero: null },
  ],
  tests: [
    test_row({}),
    test_row({ connection: "c__d", outcome: "FC full: c__d", q: 0.6, p: 0.4, estimate: 0.1 }),
    test_row({ question: "q3_detrusor_overactivity", exposure: "DO present vs DO absent", outcome: "total lesion volume",
      test: "mann_whitney", n_group1: 2, n_group0: 18, status: "skipped: needs 6 in total and 3 per group",
      estimate: null, p: null, q: null, state: "", connection: "" }),
    test_row({ question: "q4_symptom_severity", outcome: "nbss_total", q: null, state: "", connection: "" }),
  ],
};

test("numbers are written out in full, to three figures", () => {
  assert.equal(num(1182.66), "1183");
  assert.equal(num(0.5934), "0.593");
  assert.equal(num(17.809), "17.8");
  assert.equal(num(7), "7");
  assert.equal(num(null), "");
  assert.equal(pvalue(0.0004), "<0.001");
  assert.equal(pvalue(0.0164), "0.016");
});

test("column names become readable labels", () => {
  assert.equal(label("lesion_volume_norm_ml"), "Lesion volume, adjusted for head size (mL)");
  assert.equal(label("tract_ATR_ml"), "ATR lesion volume (mL)");
  assert.equal(label("tract_SLF_norm_ml"), "SLF lesion volume, adjusted (mL)");
  assert.equal(label("nbss_total"), "NBSS total");
});

test("text from the tables is escaped before it becomes html", () => {
  assert.equal(esc('<img src=x onerror="a">'), "&lt;img src=x onerror=&quot;a&quot;&gt;");
  const nasty = { ...summary, tests: [test_row({ connection: "<script>x</script>" })] };
  const html = reportHtml(nasty, measurements);
  assert.ok(!html.includes("<script>"));
  assert.ok(html.includes("&lt;script&gt;"));
  const badId = reportHtml(null, [{ ...measurements[0], id: '"><b>' }]);
  assert.ok(!badId.includes('"><b>'));
});

test("the scan table carries every measurement the questions use", () => {
  const { head, rows } = scanRows(measurements);
  assert.deepEqual(head, ["Scan", "Lesion volume (mL)", "Adjusted for head size (mL)", "Lesions",
    "ATR (mL)", "ATR, adjusted (mL)", "SLF (mL)", "SLF, adjusted (mL)", "Head-size factor"]);
  assert.deepEqual(rows[0], ["s1", 38.51, 57.93, 14, 1.34, 2.02, 0.39, 0.58, 1.5042]);
  const text = scansAsText(measurements).split("\n");
  assert.equal(text.length, 3);
  assert.equal(text[1], "s1\t38.51\t57.93\t14\t1.34\t2.02\t0.39\t0.58\t1.5042");
  assert.equal(text[2], "s2\t2.21\t\t13\t\t\t\t\t");          // missing values stay empty, not "null"
});

test("one scan's measurements as text", () => {
  const text = scanAsText({ id: "s1", stats: measurements[0] });
  assert.ok(text.startsWith("Scan\ts1\nLesion volume (mL)\t38.51\n"));
  assert.ok(text.includes("ATR lesion volume, adjusted (mL)\t2.02"));
  assert.ok(text.includes("Number of lesions\t14"));
});

test("the page has the scans, the cohort, and a section per question that ran", () => {
  const html = reportHtml(summary, measurements);
  for (const piece of ["Each scan", "Cohort", "Question 1.", "Question 3.", "Question 4."]) {
    assert.ok(html.includes(piece), piece);
  }
  assert.ok(!html.includes("Question 2.") && !html.includes("Question 5."));
  assert.ok(html.includes('data-scan="s1"'));
  assert.ok(html.includes("Connections with q below 0.05"));
  assert.ok(html.includes("a__b") && !html.includes(">c__d<"));   // only the one that passed
  assert.ok(html.includes("withheld, group too small"));
  assert.ok(html.includes("skipped: needs 6 in total"));
  assert.ok(!html.includes("fc__full__a__b"));                     // FC columns stay out of the cohort table
  assert.ok(!html.includes("Fake data"));
});

test("fake tables are announced, and no summary is said plainly", () => {
  assert.ok(reportHtml({ ...summary, fake_data: true }, measurements).includes("Fake data."));
  const empty = reportHtml(null, measurements);
  assert.ok(empty.includes("Each scan") && empty.includes("no group results yet"));
  assert.equal(reportHtml(null, []).includes("Each scan"), false);
});

test("group results from real tables block the deploy, fake ones do not", () => {
  const dir = mkdtempSync(join(tmpdir(), "viewer-"));
  const data = join(dir, "data");
  mkdirSync(join(data, "patient01"), { recursive: true });
  const allow = join(dir, "public-subjects.json");
  writeFileSync(allow, JSON.stringify({ subjects: { patient01: "public" } }));
  writeFileSync(join(data, "index.json"), JSON.stringify({ subjects: ["patient01"], measurements: [{ id: "patient01" }] }));
  assert.equal(check(data, allow).realResults, false);
  writeFileSync(join(data, "summary.json"), JSON.stringify({ fake_data: true }));
  assert.equal(check(data, allow).realResults, false);
  writeFileSync(join(data, "summary.json"), JSON.stringify({ fake_data: false }));
  assert.equal(check(data, allow).realResults, true);
  writeFileSync(join(data, "summary.json"), JSON.stringify({}));
  assert.equal(check(data, allow).realResults, true);
  // a scan named only in the measurements still has to be on the list
  writeFileSync(join(data, "index.json"), JSON.stringify({ subjects: [], measurements: [{ id: "sub-x" }] }));
  assert.deepEqual(check(data, allow).blocked, ["sub-x"]);
});
