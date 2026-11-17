import "@fontsource/ibm-plex-sans/400.css";
import "@fontsource/ibm-plex-sans/500.css";
import "@fontsource/ibm-plex-sans/600.css";
import "@fontsource/ibm-plex-sans-condensed/400.css";
import "@fontsource/ibm-plex-sans-condensed/500.css";
import "./style.css";

import {
  VIEWS, clamp, formatMl, screenToVoxel, sliceProfile, voxelToMm, voxelToScreen,
} from "./geometry.mjs";
import { SliceRenderer } from "./renderer.js";
import { reportHtml, scanAsText, scansAsText } from "./report.mjs";
import { disposeSubject, loadIndex, loadSubject, loadSummary } from "./volume.js";

const $ = (selector) => document.querySelector(selector);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const NARROW = 900;
const DEFAULTS = { level: 0.5, width: 0.5 };
// the contrast slider runs from flat (the whole grey range on screen) to steep
const widthFromSlider = (value) => 1.5 - value;
// which way each axis runs, low index first
const AXIS_ENDS = { 0: ["L", "R"], 1: ["P", "A"], 2: ["I", "S"] };

const state = {
  voxel: [0, 0, 0],
  level: DEFAULTS.level,
  width: widthFromSlider(DEFAULTS.width),
  opacity: 0.6,
  showLesion: true,
  showReference: false,
  showTracts: false,
  outline: false,
  layout: "all",
  active: "axial",   // the view the slice scrubber drives
  lesion: -1,
};

const app = $("#app");
const panesEl = $("#panes");
const panes = Object.fromEntries(
  [...document.querySelectorAll(".pane")].map((el) => [el.dataset.view, el]),
);
const track = $("#profile");
const trackCanvas = track.querySelector("canvas");
const notice = $("#notice");

let renderer = null;
let subject = null;
let profiles = {};
let loadToken = 0;
let measurements = [];
let summary = null;

// ---- drawing -------------------------------------------------------------------

// draw once on the next animation frame, however many things changed before it.
// a tab in the background gets no animation frames, so a short timer backs it up
let scheduled = false;

function requestDraw() {
  if (scheduled) return;
  scheduled = true;
  const run = () => {
    if (!scheduled) return;
    scheduled = false;
    if (!renderer || !subject) return;
    renderer.draw(panes, state);
    drawOverlays();
    drawScrubber();
    drawHere();
  };
  requestAnimationFrame(run);
  setTimeout(run, 150);
}

function drawOverlays() {
  const dims = subject.meta.dims;
  for (const [name, pane] of Object.entries(panes)) {
    const view = VIEWS[name];
    const rect = renderer.sliceRect(name, pane);
    const at = voxelToScreen(name, state.voxel, dims);
    const cx = rect.x + at.fx * rect.w;
    const cy = rect.y + at.fy * rect.h;
    pane.style.setProperty("--cx", `${cx}px`);
    pane.style.setProperty("--cy", `${cy}px`);
    pane.querySelector(".cross--h").style.top = `${cy}px`;
    pane.querySelector(".cross--v").style.left = `${cx}px`;
    pane.querySelector(".pane__slice").textContent = `${state.voxel[view.axis] + 1} of ${dims[view.axis]}`;
    pane.classList.toggle("is-active", name === state.active);
  }
}

function lesionProfile(axis) {
  if (!profiles[axis]) profiles[axis] = sliceProfile(subject.bytes.lesion, subject.meta.dims, axis);
  return profiles[axis];
}

// the scrubber's track is a small chart: one bar per slice, as tall as the lesion in it
function drawScrubber() {
  const axis = VIEWS[state.active].axis;
  const counts = lesionProfile(axis);
  const n = counts.length;
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const w = trackCanvas.clientWidth;
  const h = trackCanvas.clientHeight;
  if (trackCanvas.width !== Math.round(w * ratio) || trackCanvas.height !== Math.round(h * ratio)) {
    trackCanvas.width = Math.round(w * ratio);
    trackCanvas.height = Math.round(h * ratio);
  }
  const ctx = trackCanvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const column = w / n;
  const base = h - 5;
  ctx.fillStyle = css("--line-strong");
  ctx.fillRect(0, base, w, 1);

  let peak = 1;
  for (let i = 0; i < n; i++) peak = Math.max(peak, counts[i]);
  ctx.fillStyle = css("--lesion");
  for (let i = 0; i < n; i++) {
    if (!counts[i]) continue;
    // square-root scale, so a slice holding only a little lesion still shows
    const height = Math.max(2, Math.sqrt(counts[i] / peak) * (base - 3));
    ctx.fillRect(i * column, base - height, Math.max(column - 0.5, 1), height);
  }

  const x = (state.voxel[axis] + 0.5) * column;
  ctx.fillStyle = css("--cross");
  ctx.fillRect(Math.round(x) - 1, 0, 2, h);
  ctx.beginPath();
  ctx.arc(Math.round(x), base + 1, 4, 0, Math.PI * 2);
  ctx.fill();

  const [low, high] = AXIS_ENDS[axis];
  $("#scrub-low").textContent = low;
  $("#scrub-high").textContent = high;
  $("#scrub-where").innerHTML = `<b>${state.voxel[axis] + 1}</b> of ${n}`;
  track.setAttribute("aria-valuemin", "1");
  track.setAttribute("aria-valuemax", String(n));
  track.setAttribute("aria-valuenow", String(state.voxel[axis] + 1));
  track.setAttribute(
    "aria-valuetext",
    `${state.active} slice ${state.voxel[axis] + 1} of ${n}, ${counts[state.voxel[axis]] ? "lesion in this slice" : "no lesion in this slice"}`,
  );
}

function valueAt(name, voxel) {
  const [nx, ny] = subject.meta.dims;
  const bytes = subject.bytes[name];
  return bytes ? bytes[voxel[0] + nx * (voxel[1] + ny * voxel[2])] : 0;
}

function drawHere() {
  const { dims, voxel_mm: mm, tracts } = subject.meta;
  const [r, a, s] = voxelToMm(state.voxel, dims, mm);
  const side = (value, plus, minus) => `${value >= 0 ? plus : minus} ${Math.abs(value).toFixed(0)}`;
  const tags = [];
  if (valueAt("lesion", state.voxel)) tags.push('<span class="tag tag--lesion">lesion</span>');
  if (valueAt("reference", state.voxel)) tags.push('<span class="tag tag--plain">expert mask</span>');
  const tract = tracts?.[String(valueAt("tracts", state.voxel))];
  if (tract) tags.push(`<span class="tag tag--plain">${tract}</span>`);
  $("#here").innerHTML = [
    ["Position", `${side(r, "R", "L")}, ${side(a, "A", "P")}, ${side(s, "S", "I")} mm`],
    ["T1 brightness", `${valueAt("t1", state.voxel)} of 255`],
    ["Inside", tags.length ? tags.join("") : "no overlay"],
  ].map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
}

// ---- moving around -------------------------------------------------------------

function moveTo(voxel) {
  if (!voxel) return;
  const dims = subject.meta.dims;
  state.voxel = voxel.map((v, i) => clamp(Math.round(v), 0, dims[i] - 1));
  requestDraw();
}

function step(axis, amount) {
  const next = state.voxel.slice();
  next[axis] += amount;
  moveTo(next);
}

function setActive(name) {
  if (state.active === name) return;
  state.active = name;
  requestDraw();
}

function goToLesion(index) {
  const lesions = subject.meta.lesions;
  if (!lesions.length) return;
  state.lesion = (index + lesions.length) % lesions.length;
  moveTo(lesions[state.lesion].voxel);
  document.querySelectorAll("#lesions button").forEach((button, i) => {
    const current = i === state.lesion;
    button.setAttribute("aria-current", String(current));
    if (current) button.scrollIntoView({ block: "nearest" });
  });
}

function pointInPane(name, event) {
  const pane = panes[name];
  const box = pane.getBoundingClientRect();
  const rect = renderer.sliceRect(name, pane);
  if (rect.w < 1) return null;
  const fx = clamp((event.clientX - box.left - rect.x) / rect.w, 0, 1);
  const fy = clamp((event.clientY - box.top - rect.y) / rect.h, 0, 1);
  return screenToVoxel(name, fx, fy, state.voxel, subject.meta.dims);
}

// keep receiving the pointer while it is dragged outside the element
function capture(element, event) {
  try {
    element.setPointerCapture(event.pointerId);
  } catch {
    // a pointer that is already gone cannot be captured, and the drag simply ends early
  }
}

function bindPane(name, pane) {
  const view = VIEWS[name];
  let dragging = false;

  pane.addEventListener("pointerdown", (event) => {
    if (!subject) return;
    dragging = true;
    capture(pane, event);
    setActive(name);
    moveTo(pointInPane(name, event));
  });
  pane.addEventListener("pointermove", (event) => {
    if (dragging && subject) moveTo(pointInPane(name, event));
  });
  const stop = () => { dragging = false; };
  pane.addEventListener("pointerup", stop);
  pane.addEventListener("pointercancel", stop);
  pane.addEventListener("focus", () => setActive(name));

  pane.addEventListener("wheel", (event) => {
    if (!subject) return;
    event.preventDefault();
    setActive(name);
    step(view.axis, event.deltaY < 0 ? 1 : -1);
  }, { passive: false });

  pane.addEventListener("keydown", (event) => {
    if (!subject) return;
    const inPlane = {
      ArrowLeft: [view.h, view.flipH ? 1 : -1], ArrowRight: [view.h, view.flipH ? -1 : 1],
      ArrowUp: [view.v, 1], ArrowDown: [view.v, -1],
    };
    const through = { ArrowUp: 1, ArrowDown: -1, PageUp: 10, PageDown: -10 };
    if (event.shiftKey && inPlane[event.key]) step(...inPlane[event.key]);
    else if (through[event.key]) step(view.axis, through[event.key]);
    else if (event.key === "Home") step(view.axis, -1e6);
    else if (event.key === "End") step(view.axis, 1e6);
    else return;
    event.preventDefault();
  });
}

function bindScrubber() {
  const axis = () => VIEWS[state.active].axis;
  let dragging = false;
  const seek = (event) => {
    const box = trackCanvas.getBoundingClientRect();
    const fraction = clamp((event.clientX - box.left) / box.width, 0, 1);
    const next = state.voxel.slice();
    next[axis()] = Math.floor(fraction * subject.meta.dims[axis()]);
    moveTo(next);
  };
  track.addEventListener("pointerdown", (event) => {
    if (!subject) return;
    dragging = true;
    capture(track, event);
    seek(event);
  });
  track.addEventListener("pointermove", (event) => { if (dragging && subject) seek(event); });
  track.addEventListener("pointerup", () => { dragging = false; });
  track.addEventListener("pointercancel", () => { dragging = false; });
  track.addEventListener("keydown", (event) => {
    if (!subject) return;
    const amount = {
      ArrowRight: 1, ArrowUp: 1, ArrowLeft: -1, ArrowDown: -1,
      PageUp: 10, PageDown: -10, Home: -1e6, End: 1e6,
    }[event.key];
    if (!amount) return;
    event.preventDefault();
    step(axis(), amount);
  });
  $("#slice-back").addEventListener("click", () => subject && step(axis(), -1));
  $("#slice-forward").addEventListener("click", () => subject && step(axis(), 1));
}

// ---- layout: which views, and the two side panels ---------------------------------

function setLayout(layout) {
  state.layout = layout;
  panesEl.dataset.layout = layout;
  if (layout !== "all") state.active = layout;
  requestDraw();
}

function setPanel(side, open) {
  const hidden = `${side}-hidden`;
  app.classList.toggle(hidden, !open);
  $(`#toggle-${side}`).setAttribute("aria-expanded", String(open));
  // the canvas has to follow the stage as the panel slides
  const start = performance.now();
  const follow = () => {
    requestDraw();
    if (performance.now() - start < 320) setTimeout(follow, 40);
  };
  follow();
}

function bindLayout() {
  document.querySelectorAll('input[name="layout"]').forEach((radio) => {
    radio.addEventListener("change", () => { if (radio.checked) setLayout(radio.value); });
  });
  for (const side of ["controls", "readout"]) {
    $(`#toggle-${side}`).addEventListener("click", () => {
      const open = app.classList.contains(`${side}-hidden`);
      // on a narrow screen the panels are drawers, and only one is out at a time
      if (open && window.innerWidth <= NARROW) setPanel(side === "controls" ? "readout" : "controls", false);
      setPanel(side, open);
    });
  }
  const embedded = new URL(window.location.href).searchParams.get("embed") === "1";
  if (embedded || window.innerWidth <= NARROW) {
    setPanel("controls", false);
    setPanel("readout", false);
  }
  // from here on the panels may slide
  setTimeout(() => app.classList.remove("is-starting"), 60);
}

// ---- scans or group results ---------------------------------------------------------

async function copyText(text, status) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    // older browsers, and pages not served over https: select and copy the long way
    const area = document.createElement("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.append(area);
    area.select();
    document.execCommand("copy");
    area.remove();
  }
  status.textContent = "Copied";
  setTimeout(() => { status.textContent = ""; }, 2000);
}

function setMode(mode) {
  const results = mode === "results";
  app.classList.toggle("is-results", results);
  $("#report").hidden = !results;
  document.querySelector(`input[name="mode"][value="${mode}"]`).checked = true;
  if (!results) requestDraw();
}

function fillReport() {
  const run = summary?.run;
  $("#report-meta").textContent = run
    ? `mslb ${run.versions?.mslb ?? ""}, MindGlide ${run.versions?.mindglide ?? ""}, finished ${(run.finished ?? "").slice(0, 10)}`
    : "";
  $("#report-body").innerHTML = reportHtml(summary, measurements);
  $("#copy-scans").hidden = !measurements.length;
  $("#copy-summary").hidden = !summary?.markdown;
}

function bindReport() {
  document.querySelectorAll('input[name="mode"]').forEach((radio) => {
    radio.addEventListener("change", () => { if (radio.checked) setMode(radio.value); });
  });
  $("#copy-scans").addEventListener("click", () => copyText(scansAsText(measurements), $("#report-copied")));
  $("#copy-summary").addEventListener("click", () => copyText(summary.markdown, $("#report-copied")));
  $("#copy-scan").addEventListener("click", () => {
    if (subject) copyText(scanAsText(subject.meta), $("#scan-copied"));
  });
  // a scan's name in the table opens that scan
  $("#report-body").addEventListener("click", (event) => {
    const button = event.target instanceof Element ? event.target.closest("button[data-scan]") : null;
    if (!button) return;
    $("#scan").value = button.dataset.scan;
    setMode("scans");
    show(button.dataset.scan);
  });
}

// ---- the control panel -----------------------------------------------------------

function bindControls() {
  const toggle = (id, key) => $(id).addEventListener("change", (event) => {
    state[key] = event.target.checked;
    requestDraw();
  });
  toggle("#show-lesion", "showLesion");
  toggle("#show-reference", "showReference");
  toggle("#show-tracts", "showTracts");
  $("#show-cross").addEventListener("change", (event) => {
    panesEl.classList.toggle("no-cross", !event.target.checked);
  });
  document.querySelectorAll('input[name="style"]').forEach((radio) => {
    radio.addEventListener("change", () => {
      state.outline = document.querySelector('input[name="style"]:checked').value === "outline";
      requestDraw();
    });
  });
  $("#opacity").addEventListener("input", (event) => {
    state.opacity = Number(event.target.value);
    $("#opacity-out").textContent = `${Math.round(state.opacity * 100)}%`;
    requestDraw();
  });
  const image = () => {
    state.level = 1 - Number($("#level").value);
    state.width = widthFromSlider(Number($("#width").value));
    $("#level-out").textContent = Math.round(Number($("#level").value) * 100);
    $("#width-out").textContent = Math.round(Number($("#width").value) * 100);
    requestDraw();
  };
  $("#level").addEventListener("input", image);
  $("#width").addEventListener("input", image);
  $("#reset-image").addEventListener("click", () => {
    $("#level").value = DEFAULTS.level;
    $("#width").value = DEFAULTS.width;
    image();
  });
  $("#prev-lesion").addEventListener("click", () => goToLesion(state.lesion - 1));
  $("#next-lesion").addEventListener("click", () => goToLesion(state.lesion + 1));
  $("#scan").addEventListener("change", (event) => show(event.target.value));

  document.addEventListener("keydown", (event) => {
    if (!subject || event.ctrlKey || event.metaKey || event.altKey) return;
    if (app.classList.contains("is-results")) return;
    if (event.target instanceof Element && event.target.closest("input, select, textarea")) return;
    const key = event.key.toLowerCase();
    if (key === "[") goToLesion(state.lesion - 1);
    else if (key === "]") goToLesion(state.lesion + 1);
    else if (key === "l") $("#show-lesion").click();
    else if (key === "o") {
      const other = document.querySelector('input[name="style"]:not(:checked)');
      other.checked = true;
      other.dispatchEvent(new Event("change"));
    }
  });
}

// ---- the measurements panel ---------------------------------------------------------

function known(value) {
  return value !== null && value !== undefined && !Number.isNaN(value);
}

function fillReadout(meta) {
  const s = meta.stats;
  const lesions = meta.lesions;

  // the pipeline's number when it is there. without it, add up what the viewer can see
  const fromList = lesions.reduce((sum, lesion) => sum + lesion.ml, 0);
  const volume = known(s.lesion_volume_ml) ? s.lesion_volume_ml : fromList;
  $("#hero-value").textContent = volume < 10 ? volume.toFixed(2) : volume.toFixed(1);
  $("#hero-unit").textContent = "mL";
  $("#hero-note").innerHTML = known(s.lesion_volume_norm_ml)
    ? `<b>${formatMl(s.lesion_volume_norm_ml)}</b> once adjusted for head size`
    : known(s.lesion_volume_ml) ? "" : "Measured on the viewer's 1 mm copy of the scan.";

  const facts = [["Lesions", known(s.lesion_count) ? s.lesion_count : lesions.length]];
  if (known(s.head_scale)) facts.push(["Head-size factor", s.head_scale.toFixed(2)]);
  $("#facts").innerHTML = facts.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");

  const colours = [css("--tract-1"), css("--tract-2")];
  const loads = Object.entries(s.tracts ?? {}).filter(([, load]) => known(load.ml));
  $("#tracts-group").hidden = !loads.length;
  $("#meters").innerHTML = loads.map(([name, load], i) => `
    <li>
      <span class="dot" style="background:${colours[i % colours.length]}" aria-hidden="true"></span>
      <span class="name">${name}</span>
      <span class="value">${formatMl(load.ml)}</span>
      <span class="share">${[
    known(load.norm_ml) ? `${formatMl(load.norm_ml)} adjusted for head size` : "",
    known(load.pct) ? `${load.pct.toFixed(1)}% of the tract is lesion` : "",
  ].filter(Boolean).join(", ")}</span>
    </li>`).join("");

  const list = $("#lesions");
  list.innerHTML = "";
  if (!lesions.length) {
    list.innerHTML = '<li class="lesions__none">No lesions were found on this scan.</li>';
  }
  const largest = lesions.length ? lesions[0].ml : 1;
  lesions.forEach((lesion, i) => {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.setAttribute("aria-label", `Lesion ${i + 1}, ${formatMl(lesion.ml)}`);
    // square-root scale again: most lesions are tiny beside the largest
    const share = Math.max(2, Math.sqrt(lesion.ml / largest) * 100);
    button.innerHTML = `<span class="rank">${i + 1}</span>`
      + `<span class="bar" style="--share:${share.toFixed(1)}%"></span>`
      + `<span class="ml">${formatMl(lesion.ml)}</span>`;
    button.addEventListener("click", () => goToLesion(i));
    item.append(button);
    list.append(item);
  });
  $("#prev-lesion").disabled = $("#next-lesion").disabled = !lesions.length;

  const layerRow = (row, box, key, present) => {
    $(row).hidden = !present;
    if (!present) { $(box).checked = false; state[key] = false; }
  };
  layerRow("#reference-row", "#show-reference", "showReference", Boolean(meta.layers.reference));
  layerRow("#tracts-row", "#show-tracts", "showTracts", Boolean(meta.layers.tracts));
  const names = Object.values(meta.tracts ?? {});
  $("#tracts-name").innerHTML = names.length ? `Tracts <small>${names.join(", ")}</small>` : "Tracts";
  $("#attribution").textContent = meta.attribution || "";
}

function say(html, retry) {
  notice.hidden = !html;
  notice.innerHTML = html ? `<p>${html}</p>` : "";
  if (html && retry) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = "Try again";
    button.addEventListener("click", retry);
    notice.append(button);
  }
}

async function show(id) {
  const token = ++loadToken;
  say("Loading the scan.");
  try {
    const next = await loadSubject(id);
    if (token !== loadToken) { disposeSubject(next); return; }   // a newer choice won
    disposeSubject(subject);
    subject = next;
    profiles = {};
    state.lesion = -1;
    renderer.setSubject(subject);
    fillReadout(subject.meta);
    for (const [name, pane] of Object.entries(panes)) {
      for (const [edge, letter] of Object.entries(VIEWS[name].labels)) {
        pane.querySelector(`.mark--${edge}`).textContent = letter;
      }
    }
    const lesions = subject.meta.lesions;
    moveTo(lesions.length ? lesions[0].voxel : subject.meta.dims.map((n) => n / 2));
    if (lesions.length) goToLesion(0);
    const url = new URL(window.location.href);
    url.searchParams.set("scan", id);
    window.history.replaceState(null, "", url);
    say("");
  } catch (error) {
    if (token === loadToken) say(`The scan could not be loaded. ${error.message}`, () => show(id));
  }
}

async function start() {
  try {
    renderer = new SliceRenderer($("#gl"), {
      lesion: css("--lesion"), reference: css("--reference"),
      tracts: [css("--tract-1"), css("--tract-2")],
    });
  } catch (error) {
    say("This browser cannot draw the images: it needs WebGL 2.");
    return;
  }
  Object.entries(panes).forEach(([name, pane]) => bindPane(name, pane));
  bindScrubber();
  bindControls();
  bindLayout();
  bindReport();
  new ResizeObserver(requestDraw).observe($("#stage"));
  window.addEventListener("resize", requestDraw);

  let ids = [];
  try {
    const index = await loadIndex();
    ids = index.subjects;
    measurements = index.measurements;
    summary = await loadSummary();
  } catch (error) {
    ids = [];
  }
  fillReport();
  if (!ids.length) {
    say("There are no scans to show yet. Run <code>mslb run</code>, then <code>mslb view</code>.");
    return;
  }
  const select = $("#scan");
  ids.forEach((id) => select.add(new Option(id, id)));
  const wanted = new URL(window.location.href).searchParams.get("scan");
  select.value = ids.includes(wanted) ? wanted : ids[0];
  show(select.value);
}

start();
