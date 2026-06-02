import { copyFile, mkdir } from "fs/promises";
import { resolve } from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = resolve(__filename, "..");
const projectRoot = resolve(__dirname, "..", "..");
const targetDir = resolve(__dirname, "..", "data");

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
