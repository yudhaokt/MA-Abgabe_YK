from pathlib import Path
import random
import shutil


# SETTINGS
SOURCE_DIR = Path("Datasets/RQD")
OUTPUT_DIR = Path("Datasets/RQD_remapped")

TRAIN_RATIO = 0.65
VAL_RATIO = 0.20
TEST_RATIO = 0.15

SEED = 42

SOURCE_SPLITS = ["train", "val", "test"]

CLASS_MAPPING = {
    "1": "0_excellent",
    "2": "1_good",
    "3": "2_intermediate",
    "4": "3_bad_paved",
    "5": "4_bad_unpaved",
    "6": "5_unclassified",
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# CHECK RATIOS
assert abs(
    TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0
) < 1e-6, "Train, validation, and test ratios must sum to 1."


# DELETE EXISTING OUTPUT DATASET
if OUTPUT_DIR.exists():
    print(f"Removing existing output directory: {OUTPUT_DIR}")
    shutil.rmtree(OUTPUT_DIR)


# CREATE OUTPUT FOLDERS
for split in ["train", "val", "test"]:
    for new_class in CLASS_MAPPING.values():

        folder = OUTPUT_DIR / split / new_class
        folder.mkdir(parents=True, exist_ok=True)


# RANDOM GENERATOR
random.seed(SEED)

summary = []


# COMBINE ORIGINAL TRAIN / VAL / TEST FOR EACH CLASS
# AND CREATE NEW SPLITS
for old_class, new_class in CLASS_MAPPING.items():

    print(f"\nProcessing class {old_class} -> {new_class}")

    # Collect images from original train, val, and test
    images = []

    for source_split in SOURCE_SPLITS:

        source_class_dir = SOURCE_DIR / source_split / old_class

        if not source_class_dir.exists():
            print(f"WARNING: Folder not found: {source_class_dir}")
            continue

        split_images = [
            file
            for file in source_class_dir.iterdir()
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        ]

        # Store both path and original split name
        for image in split_images:
            images.append((image, source_split))

        print(
            f"  {source_split}: "
            f"{len(split_images)} images"
        )

    # Shuffle combined dataset
    random.shuffle(images)

    total = len(images)

    # Calculate new split sizes
    n_train = int(total * TRAIN_RATIO)
    n_val = int(total * VAL_RATIO)
    n_test = total - n_train - n_val

    train_images = images[:n_train]

    val_images = images[
        n_train:n_train + n_val
    ]

    test_images = images[
        n_train + n_val:
    ]

    # Copy helper function
    def copy_images(image_list, destination_split):

        destination_dir = (
            OUTPUT_DIR
            / destination_split
            / new_class
        )

        for image_path, original_split in image_list:

            # Prefix original split to avoid filename collisions
            new_filename = (
                f"{original_split}_{image_path.name}"
            )

            destination = (
                destination_dir / new_filename
            )

            shutil.copy2(
                image_path,
                destination
            )

    # Copy images into new dataset
    copy_images(
        train_images,
        "train"
    )

    copy_images(
        val_images,
        "val"
    )

    copy_images(
        test_images,
        "test"
    )

    # Save summary
    summary.append(
        (
            old_class,
            new_class,
            total,
            len(train_images),
            len(val_images),
            len(test_images),
        )
    )


# PRINT SUMMARY
print("\n" + "=" * 70)
print("REMAPPING SUMMARY")
print("=" * 70)

print(
    f"{'Old':<8}"
    f"{'New Class':<22}"
    f"{'Total':>10}"
    f"{'Train':>10}"
    f"{'Val':>10}"
    f"{'Test':>10}"
)

print("-" * 70)

for (
    old_class,
    new_class,
    total,
    train_count,
    val_count,
    test_count,
) in summary:

    print(
        f"{old_class:<8}"
        f"{new_class:<22}"
        f"{total:>10}"
        f"{train_count:>10}"
        f"{val_count:>10}"
        f"{test_count:>10}"
    )


print("-" * 75)

total_images = sum(row[2] for row in summary)
total_train = sum(row[3] for row in summary)
total_val = sum(row[4] for row in summary)
total_test = sum(row[5] for row in summary)

print(
    f"{'TOTAL':<30}"
    f"{total_images:>10}"
    f"{total_train:>10}"
    f"{total_val:>10}"
    f"{total_test:>10}"
)

print("\nRemapping completed successfully!")
print(f"Output dataset: {OUTPUT_DIR}")