// the maths behind the three slice views, kept free of three.js and the dom so it can be tested.
//
// volumes are on a grid whose first axis runs right, second runs forward and third runs up
// (RAS), stored x fastest. a view looks along one axis and lays the other two out on screen.
// the display follows the radiological habit: on axial and coronal slices the subject's left
// is on the right of the screen.

export const VIEWS = {
  // h and v are the volume axes drawn across and up the screen. flipH mirrors the horizontal.
  axial: { axis: 2, h: 0, v: 1, flipH: true, labels: { left: "R", right: "L", top: "A", bottom: "P" } },
  coronal: { axis: 1, h: 0, v: 2, flipH: true, labels: { left: "R", right: "L", top: "S", bottom: "I" } },
  sagittal: { axis: 0, h: 1, v: 2, flipH: false, labels: { left: "P", right: "A", top: "S", bottom: "I" } },
};

export const clamp = (value, lo, hi) => Math.min(Math.max(value, lo), hi);

// the largest rectangle with the slice's physical shape that fits in a pane, centred
export function fitRect(paneW, paneH, sliceW, sliceH) {
  if (paneW <= 0 || paneH <= 0 || sliceW <= 0 || sliceH <= 0) return { x: 0, y: 0, w: 0, h: 0 };
  const scale = Math.min(paneW / sliceW, paneH / sliceH);
  const w = sliceW * scale;
  const h = sliceH * scale;
  return { x: (paneW - w) / 2, y: (paneH - h) / 2, w, h };
}

// a point in the drawn slice, as fractions from its left and top edges, to a voxel index.
// the voxel along the viewing axis comes from the current position.
export function screenToVoxel(viewName, fx, fy, voxel, dims) {
  const view = VIEWS[viewName];
  const out = voxel.slice();
  const across = view.flipH ? 1 - fx : fx;
  out[view.h] = clamp(Math.floor(across * dims[view.h]), 0, dims[view.h] - 1);
  out[view.v] = clamp(Math.floor((1 - fy) * dims[view.v]), 0, dims[view.v] - 1);
  return out;
}

// the reverse: where a voxel's centre falls in the drawn slice, as fractions from left and top
export function voxelToScreen(viewName, voxel, dims) {
  const view = VIEWS[viewName];
  const across = (voxel[view.h] + 0.5) / dims[view.h];
  return {
    fx: view.flipH ? 1 - across : across,
    fy: 1 - (voxel[view.v] + 0.5) / dims[view.v],
  };
}

// texture coordinate of a slice's centre along its axis
export const sliceCoord = (index, n) => (index + 0.5) / n;

// how many mask voxels sit in each slice along one axis. this is the lesion profile
// drawn beside a view: it shows where along the stack the lesions are.
export function sliceProfile(mask, dims, axis) {
  const [nx, ny, nz] = dims;
  const counts = new Uint32Array(dims[axis]);
  let i = 0;
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++, i++) {
        if (mask[i]) counts[axis === 0 ? x : axis === 1 ? y : z]++;
      }
    }
  }
  return counts;
}

// brightness and contrast as the two numbers the shader needs: the grey level that
// maps to black and the span that maps black to white, both on a 0 to 1 scale
export function windowRange(level, width) {
  const w = Math.max(width, 1 / 255);
  return { low: level - w / 2, span: w };
}

// right-anterior-superior position in mm, measured from the volume's centre
export function voxelToMm(voxel, dims, voxelMm) {
  return voxel.map((v, i) => (v + 0.5 - dims[i] / 2) * voxelMm);
}

export function formatMl(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "not available";
  if (value === 0) return "0 mL";
  if (value < 0.005) return "under 0.01 mL";
  return `${value < 10 ? value.toFixed(2) : value.toFixed(1)} mL`;
}
