import { copyFile, mkdir } from "fs/promises";
import { spawnSync } from "child_process";
import { resolve } from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = resolve(__filename, "..");
const projectRoot = resolve(__dirname, "..", "..");
const targetDir = resolve(__dirname, "..", "data");

// v3 site data (web/data/v3/site.json) comes from the Python pipeline.
const py = spawnSync(process.platform === "win32" ? "python" : "python3", ["-m", "flatout", "export"],
  { cwd: projectRoot, stdio: "inherit" });
if (py.status !== 0) console.warn("Skipping v3 export (python -m flatout export failed); serving existing data.");

// v2 legacy dashboard files
const files = [
  "v2_prediction_results.csv",
  "v2_full_predictions.csv",
  "v2_feature_importance.csv"
];

await mkdir(targetDir, { recursive: true });

for (const file of files) {
  const src = resolve(projectRoot, file);
  const dest = resolve(targetDir, file);
  try {
    await copyFile(src, dest);
    console.log(`Copied ${file}`);
  } catch (err) {
    console.warn(`Skipping ${file}: ${err.message}`);
  }
}

console.log("Data sync complete.");
