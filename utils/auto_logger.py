import os
import csv
import uuid
import hashlib
import sys
import signal
import atexit
from datetime import datetime

BASE_DIR = r"C:\Users\97433\Knowing_the_difference"

RUN_REGISTRY = os.path.join(BASE_DIR, "run_registry.csv")
DATASET_REGISTRY = os.path.join(BASE_DIR, "dataset_registry.csv")

CURRENT_RUN_ID = None
RUN_STATUS = "RUNNING"


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def generate_run_id():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = str(uuid.uuid4())[:8]
    return f"run_{timestamp}_{uid}"


def generate_code_id(code_filepath):
    with open(code_filepath, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()[:12]

    return f"code_{file_hash}"


def ensure_csv_exists(filepath, headers):

    if not os.path.exists(filepath):
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)


# --------------------------------------------------
# Dataset Registry
# --------------------------------------------------

def register_dataset(dataset_name,
                     train_dataset_folderpath,
                     test_dataset_folderpath,
                     val_dataset_folderpath):

    ensure_csv_exists(
        DATASET_REGISTRY,
        ["dataset_name",
         "train_dataset_folderpath",
         "test_dataset_folderpath",
         "val_dataset_folderpath"]
    )

    with open(DATASET_REGISTRY, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row["dataset_name"] == dataset_name:
                return

    with open(DATASET_REGISTRY, "a", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            dataset_name,
            train_dataset_folderpath,
            test_dataset_folderpath,
            val_dataset_folderpath
        ])


# --------------------------------------------------
# Run Status Management
# --------------------------------------------------

def update_run_status(run_id, status, error_msg=""):

    rows = []

    with open(RUN_REGISTRY, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        if row["run_id"] == run_id:
            row["status"] = status
            row["error_message"] = error_msg

    with open(RUN_REGISTRY, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "code_id",
                "code_filepath",
                "dataset_name",
                "log_text_filepath",
                "output_dir",
                "status",
                "error_message"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)


def handle_exception(exc_type, exc_value, exc_traceback):

    global CURRENT_RUN_ID

    if CURRENT_RUN_ID is None:
        return

    error_msg = str(exc_value)

    update_run_status(CURRENT_RUN_ID, "FAILED", error_msg)


def handle_interrupt(sig, frame):

    global CURRENT_RUN_ID

    if CURRENT_RUN_ID is None:
        return

    update_run_status(CURRENT_RUN_ID, "INTERRUPTED")

    sys.exit(1)


def handle_success():

    global CURRENT_RUN_ID, RUN_STATUS

    if CURRENT_RUN_ID and RUN_STATUS == "RUNNING":
        update_run_status(CURRENT_RUN_ID, "SUCCESS")


# --------------------------------------------------
# Run Logger
# --------------------------------------------------

def log_run(code_filepath, dataset_name, base_output_dir="runs"):

    global CURRENT_RUN_ID

    ensure_csv_exists(
        RUN_REGISTRY,
        [
            "run_id",
            "code_id",
            "code_filepath",
            "dataset_name",
            "log_text_filepath",
            "output_dir",
            "status",
            "error_message"
        ]
    )

    run_id = generate_run_id()
    code_id = generate_code_id(code_filepath)

    output_dir = os.path.join(base_output_dir, run_id)
    os.makedirs(output_dir, exist_ok=True)

    log_text_filepath = os.path.join(output_dir, "run_log.txt")

    open(log_text_filepath, "w").close()

    with open(RUN_REGISTRY, "a", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            run_id,
            code_id,
            code_filepath,
            dataset_name,
            log_text_filepath,
            output_dir,
            "RUNNING",
            ""
        ])

    CURRENT_RUN_ID = run_id

    # Register global handlers
    sys.excepthook = handle_exception
    signal.signal(signal.SIGINT, handle_interrupt)
    atexit.register(handle_success)

    return run_id, output_dir, log_text_filepath