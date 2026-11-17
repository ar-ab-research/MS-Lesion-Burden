// runs before npm run deploy. the site on github pages is public and is built from
// whatever is in public/data, so a scan only ships when it is named in
// public-subjects.json, and group results only ship when they were computed on fake
// tables. anything else stops the deploy before it builds.
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");

const readJson = (path) => JSON.parse(readFileSync(path, "utf8"));

export function check(dataDir, allowFile) {
  const allowed = readJson(allowFile).subjects ?? {};
  const folders = existsSync(dataDir)
    ? readdirSync(dataDir, { withFileTypes: true }).filter((d) => d.isDirectory()).map((d) => d.name)
    : [];
  const indexFile = join(dataDir, "index.json");
  const index = existsSync(indexFile) ? readJson(indexFile) : {};
  const listed = [...(index.subjects ?? []), ...(index.measurements ?? []).map((m) => m.id)];
  // a folder counts even when the menu does not list it: its files would still be published
  const present = [...new Set([...folders, ...listed])].sort();
  const summaryFile = join(dataDir, "summary.json");
  // results from real clinical data are not for a public page, whatever the scans are
  const realResults = existsSync(summaryFile) && readJson(summaryFile).fake_data !== true;
  return { present, blocked: present.filter((s) => !(s in allowed)), realResults };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const { present, blocked, realResults } = check(
    join(root, "public", "data"), join(root, "public-subjects.json"),
  );
  if (!present.length) {
    console.error("no scans in public/data. export the public ones first, see the readme");
    process.exit(1);
  }
  if (blocked.length) {
    console.error(`not deploying. these scans are not on public-subjects.json: ${blocked.join(", ")}`);
    console.error("delete viewer/public/data and export only the public scans again");
    process.exit(1);
  }
  if (realResults) {
    console.error("not deploying. public/data/summary.json holds group results that were not computed on fake tables");
    process.exit(1);
  }
  console.log(`public scans ok: ${present.join(", ")}`);
}
