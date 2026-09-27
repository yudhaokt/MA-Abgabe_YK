from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import numpy as np

# Model path
MODEL_PATH = r"Results/YOLO11x_smoothness-cls_best.pt"
## or alternative below
# MODEL_PATH = r"ModelTraining/RQD-remapped_YOLO11x/RQD-remapped_train/weights/best.pt"


INPUT_DIR = Path("GSV_Downloader/images")

OUTPUT_CSV = Path(
    "ModelApplication/road_feature_classification_log.csv"
)

IMGSZ = 224
DEVICE = 0

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def get_image_files(folder: Path):
    """
    Return all supported images inside one road-feature folder.
    """
    return sorted([
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ])


def class_names_from_model(model):
    """
    Return class names in the same order as the probability
    vector produced by the YOLO classification model.
    """
    names = model.names

    if isinstance(names, dict):
        return [names[i] for i in sorted(names.keys())]

    return list(names)


def classify_road_feature(
    model: YOLO,
    road_folder: Path,
    class_names
):
    """
    Classify every Street View image belonging to one road feature.

    The class probability vectors from all successfully classified
    images are averaged. The road-level smoothness class is defined
    as the class with the highest average probability.
    """

    osmid = road_folder.name

    image_files = get_image_files(road_folder)

    if not image_files:
        print(f"[SKIP] {osmid}: no images found.")
        return None


    print()
    print("=" * 60)
    print(f"Processing road feature: {osmid}")
    print(f"Images found: {len(image_files)}")
    print("=" * 60)


    probability_vectors = []

    # Optional information useful for reviewing the result
    individual_predictions = []


    for image_path in image_files:

        try:
            result = model.predict(
                source=str(image_path),
                imgsz=IMGSZ,
                device=DEVICE,
                verbose=False
            )[0]

            probabilities = (
                result.probs.data
                .detach()
                .cpu()
                .numpy()
                .astype(float)
            )

            probability_vectors.append(probabilities)


            image_class_id = int(
                np.argmax(probabilities)
            )

            image_class_name = class_names[
                image_class_id
            ]

            image_confidence = float(
                probabilities[image_class_id]
            )


            individual_predictions.append(
                f"{image_path.name}:{image_class_name}"
                f"({image_confidence:.4f})"
            )


            print(
                f"{image_path.name} -> "
                f"{image_class_name} "
                f"({image_confidence:.4f})"
            )


        except Exception as e:

            print(
                f"[ERROR] Could not classify "
                f"{image_path.name}: {e}"
            )


    # No successful classifications
    if not probability_vectors:

        print(
            f"[SKIP] {osmid}: "
            "no image could be classified."
        )

        return None


    # Average class probabilities across all images
    probability_matrix = np.stack(
        probability_vectors,
        axis=0
    )

    average_probabilities = probability_matrix.mean(
        axis=0
    )


    predicted_class_id = int(
        np.argmax(average_probabilities)
    )

    predicted_class = class_names[
        predicted_class_id
    ]

    road_confidence = float(
        average_probabilities[
            predicted_class_id
        ]
    )


    # Build road-level CSV row
    row = {
        "osmid": osmid,
        "images_available": len(image_files),
        "images_classified": len(probability_vectors),
        "predicted_smoothness": predicted_class,
        "prediction_confidence": road_confidence,
    }


    # Store the average probability of every class.
    for class_name, probability in zip(
        class_names,
        average_probabilities
    ):

        column_name = (
            "avg_prob_"
            + str(class_name)
            .strip()
            .lower()
            .replace(" ", "_")
        )

        row[column_name] = float(
            probability
        )

    row["individual_predictions"] = " | ".join(
        individual_predictions
    )


    print(
        f"Road-level prediction: "
        f"{predicted_class} "
        f"({road_confidence:.4f})"
    )

    print("Average class probabilities:")

    for class_name, probability in zip(
        class_names,
        average_probabilities
    ):

        print(
            f"  {class_name}: "
            f"{probability:.4f}"
        )


    return row


# MAIN
def main():

    if not INPUT_DIR.exists():

        raise FileNotFoundError(
            f"Input directory not found: {INPUT_DIR}"
        )


    # Load trained model 
    print(f"Loading model: {MODEL_PATH}")

    model = YOLO(
        MODEL_PATH
    )

    class_names = class_names_from_model(
        model
    )


    print()
    print("Model classes:")

    for i, name in enumerate(
        class_names
    ):

        print(
            f"  {i}: {name}"
        )


    # Find road-feature folders
    road_folders = sorted([
        path
        for path in INPUT_DIR.iterdir()
        if path.is_dir()
    ])


    if not road_folders:

        print(
            f"No road-feature folders found "
            f"in {INPUT_DIR}"
        )

        return


    print()
    print(
        f"Road features found: "
        f"{len(road_folders)}"
    )


    # Classify each road feature
    road_results = []


    for road_folder in road_folders:

        row = classify_road_feature(
            model=model,
            road_folder=road_folder,
            class_names=class_names
        )

        if row is not None:

            road_results.append(
                row
            )


    # Save road-level classification log
    if not road_results:

        print()
        print(
            "No road-feature classifications "
            "were produced."
        )

        return


    result_df = pd.DataFrame(
        road_results
    )


    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    result_df.to_csv(
        OUTPUT_CSV,
        index=False
    )


    print()
    print("=" * 60)
    print("Classification finished.")
    print(
        f"Road features classified: "
        f"{len(result_df)}"
    )
    print(
        f"Saved road-level log: "
        f"{OUTPUT_CSV}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
