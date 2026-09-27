import pandas as pd
import re
import random
import shutil
from pathlib import Path


# SETTINGS
## logfile of classified smoothness
LOG_FILE = Path("ModelApplication/road_feature_classification_log.csv")

# location of downloaded gsv images
IMAGE_ROOT = Path("GSV_Downloader/images")

OUTPUT_DIR = Path("SHAP_Analysis/xai_selected_images")
OUTPUT_CSV = OUTPUT_DIR / "xai_selected_images.csv"

IMAGES_PER_CLASS = 2
RANDOM_SEED = 42

CLASSES = [
    "0_excellent",
    "1_good",
    "2_intermediate",
    "3_bad_paved",
    "4_bad_unpaved",
    "5_unclassified",
]


# READ CLASSIFICATION LOG
df = pd.read_csv(LOG_FILE, dtype={"osmid": str})


# EXTRACT INDIVIDUAL IMAGE PREDICTIONS
records = []

# image.jpg:2_intermediate(0.6164)
pattern = re.compile(
    r"(.+?):([0-5]_[A-Za-z_]+)\(([0-9.]+)\)"
)

for _, row in df.iterrows():

    osmid = str(row["osmid"])

    predictions = str(row["individual_predictions"]).split(" | ")

    for prediction in predictions:

        match = pattern.fullmatch(prediction.strip())

        if not match:
            print(f"Could not parse: {prediction}")
            continue

        filename = match.group(1)
        predicted_class = match.group(2)
        confidence = float(match.group(3))

        records.append({
            "osmid": osmid,
            "filename": filename,
            "predicted_class": predicted_class,
            "confidence": confidence,
        })


images_df = pd.DataFrame(records)

print("\nAvailable images per predicted class:")
print(images_df["predicted_class"].value_counts())


# RANDOM SELECTION
random.seed(RANDOM_SEED)

selected = []
used_osmids = set()


class_order = CLASSES.copy()
random.shuffle(class_order)

for class_name in class_order:

    candidates = images_df[
        images_df["predicted_class"] == class_name
    ].copy()

    # Remove roads already represented in another class
    candidates = candidates[
        ~candidates["osmid"].isin(used_osmids)
    ]

    # Randomize candidate order
    candidates = candidates.sample(
        frac=1,
        random_state=RANDOM_SEED
    )

    class_selected = []

    for _, candidate in candidates.iterrows():

        osmid = candidate["osmid"]

        # only one image from each OSM road feature
        if osmid in used_osmids:
            continue

        class_selected.append(candidate)
        used_osmids.add(osmid)

        if len(class_selected) == IMAGES_PER_CLASS:
            break

    selected.extend(class_selected)

    print(
        f"{class_name}: "
        f"{len(class_selected)}/{IMAGES_PER_CLASS} images selected"
    )


selected_df = pd.DataFrame(selected)


# CREATE OUTPUT FOLDERS
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for class_name in CLASSES:
    (OUTPUT_DIR / class_name).mkdir(
        parents=True,
        exist_ok=True
    )


# FIND AND COPY IMAGES
output_records = []

for _, row in selected_df.iterrows():

    filename = row["filename"]
    class_name = row["predicted_class"]

    matches = list(IMAGE_ROOT.rglob(filename))

    if not matches:
        print(f"Image not found: {filename}")
        continue

    source = matches[0]

    destination = (
        OUTPUT_DIR
        / class_name
        / filename
    )

    shutil.copy2(source, destination)

    output_records.append({
        "osmid": row["osmid"],
        "filename": filename,
        "predicted_class": class_name,
        "confidence": row["confidence"],
        "source_path": str(source),
        "xai_path": str(destination),
    })


# SAVE XAI SELECTION LOG
output_df = pd.DataFrame(output_records)

output_df = output_df.sort_values(
    ["predicted_class", "osmid"]
)

output_df.to_csv(
    OUTPUT_CSV,
    index=False
)


print("\n========================================")
print("XAI IMAGE SELECTION COMPLETE")
print("========================================")

print(f"\nTotal images selected: {len(output_df)}")

print("\nImages per class:")
print(output_df["predicted_class"].value_counts().sort_index())

print(
    "\nUnique OSM IDs:",
    output_df["osmid"].nunique()
)

print(f"\nSelection log saved to:\n{OUTPUT_CSV}")
print(f"\nImages saved to:\n{OUTPUT_DIR}")