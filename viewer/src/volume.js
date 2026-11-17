// loads one subject's volumes and turns them into 3d textures
import * as THREE from "three";

const DATA = "./data";

async function fetchOk(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url} could not be loaded (${response.status})`);
  return response;
}

// the list of scans, with each scan's measurements
export async function loadIndex() {
  const index = await (await fetchOk(`${DATA}/index.json`)).json();
  return { subjects: index.subjects ?? [], measurements: index.measurements ?? [] };
}

// the group summary, or null when the pipeline has not written one
export async function loadSummary() {
  const response = await fetch(`${DATA}/summary.json`);
  if (!response.ok) return null;
  try {
    return await response.json();
  } catch {
    return null;
  }
}

// the volumes are gzipped on disk. some servers undo that on the way out and some do not,
// so look at the first two bytes rather than trusting the file name
async function fetchBytes(url) {
  const buffer = await (await fetchOk(url)).arrayBuffer();
  const head = new Uint8Array(buffer, 0, Math.min(2, buffer.byteLength));
  if (head[0] !== 0x1f || head[1] !== 0x8b) return new Uint8Array(buffer);
  const stream = new Blob([buffer]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

function texture(bytes, dims, smooth) {
  const tex = new THREE.Data3DTexture(bytes, dims[0], dims[1], dims[2]);
  tex.format = THREE.RedFormat;
  tex.type = THREE.UnsignedByteType;
  tex.minFilter = tex.magFilter = smooth ? THREE.LinearFilter : THREE.NearestFilter;
  tex.unpackAlignment = 1;
  tex.needsUpdate = true;
  return tex;
}

export async function loadSubject(id) {
  const base = `${DATA}/${encodeURIComponent(id)}`;
  const meta = await (await fetchOk(`${base}/meta.json`)).json();
  const expected = meta.dims[0] * meta.dims[1] * meta.dims[2];
  const names = Object.keys(meta.layers);
  const loaded = await Promise.all(names.map((name) => fetchBytes(`${base}/${meta.layers[name]}`)));
  const bytes = {};
  const textures = {};
  names.forEach((name, i) => {
    if (loaded[i].length !== expected) {
      throw new Error(`${id}: the ${name} volume has ${loaded[i].length} voxels, expected ${expected}`);
    }
    bytes[name] = loaded[i];
    textures[name] = texture(loaded[i], meta.dims, name === "t1");
  });
  return { meta, bytes, textures };
}

export function disposeSubject(subject) {
  if (!subject) return;
  Object.values(subject.textures).forEach((tex) => tex.dispose());
}

// a 1x1x1 stand-in, so the shader always has something bound for a missing layer
export function emptyTexture() {
  return texture(new Uint8Array(1), [1, 1, 1], false);
}
