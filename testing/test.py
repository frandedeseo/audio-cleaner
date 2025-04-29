import os
import json
import time
import logging
import requests
from collections import defaultdict

# Configuration
SERVER_URL = "http://127.0.0.1:8000/evaluar-lectura"
TEST_JSON = os.path.join(os.path.dirname(__file__), "result.json")
AUDIO_DIR = os.path.join(os.path.dirname(__file__), "test-audios")
# Delay between requests in seconds
dELAY = 2

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def load_ground_truth(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def evaluate():
    gt_list = load_ground_truth(TEST_JSON)
    # Counters
    total_files = 0
    # per-criterion counts
    correct_counts = defaultdict(int)
    # total comparisons per criterion
    total_counts = defaultdict(int)

    # Store detailed logs
    details = []

    for entry in gt_list:
        total_files += 1
        audio_file = entry["audio"]
        true_labels = entry["output"]
        # if your JSON has text, adjust here; otherwise load from separate mapping
        text = entry.get("text")
        if text is None:
            logger.error(f"No 'text' field for {audio_file}, skipping.")
            continue

        audio_path = os.path.join(AUDIO_DIR, audio_file)
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            continue

        # Prepare request
        files = {"audio": open(audio_path, "rb")}
        data = {"text": text}

        try:
            logger.info(f"Sending {audio_file} to server...")
            resp = requests.post(SERVER_URL, data=data, files=files)
            resp.raise_for_status()
            result = resp.json()
        except Exception as e:
            logger.exception(f"Request failed for {audio_file}: {e}")
            continue
        finally:
            files["audio"].close()

        # Extract predicted labels
        pred_labels = {}
        if "error" in result:
            logger.warning(f"Server returned error for {audio_file}: {result.get('error')}")
        else:
            for crit, obj in result.items():
                # each obj is {"nivel": ..., ...}
                pred_labels[crit] = obj.get("nivel")

        # Compare
        file_correct = 0
        file_total = 0
        for crit, true_lvl in true_labels.items():
            pred_lvl = pred_labels.get(crit)
            total_counts[crit] += 1
            if pred_lvl == true_lvl:
                correct_counts[crit] += 1
                file_correct += 1
            file_total += 1
            logger.info(f"{audio_file} | {crit}: true={true_lvl}, pred={pred_lvl}")

        acc = file_correct / file_total if file_total else 0
        details.append({
            "audio": audio_file,
            "accuracy": acc,
            "predicted": pred_labels,
            "true": true_labels
        })
        logger.info(f"File {audio_file} accuracy: {acc:.2%}")

        # wait before next request
        time.sleep(dELAY)

    # Compute overall
    logger.info("\n=== SUMMARY ===")
    # per-criterion accuracy
    for crit in total_counts:
        acc = correct_counts[crit] / total_counts[crit] if total_counts[crit] else 0
        logger.info(f"{crit} accuracy: {acc:.2%} ({correct_counts[crit]}/{total_counts[crit]})")
    # average
    avg_acc = sum(correct_counts.values()) / sum(total_counts.values()) if sum(total_counts.values()) else 0
    logger.info(f"Average accuracy: {avg_acc:.2%}")

    # Optionally write details to file
    with open("results_log.json", "w", encoding="utf-8") as out_f:
        json.dump(details, out_f, ensure_ascii=False, indent=2)
    logger.info("Detailed results written to results_log.json")


if __name__ == "__main__":
    evaluate()
