from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


# CONFIGURATION
# Trained YOLO classification model
MODEL_PATH = r"Results/YOLO11x_smoothness-cls_best.pt"
## or alternative below
# MODEL_PATH = r"ModelTraining/RQD-remapped_YOLO11x/RQD-remapped_train/weights/best.pt"


INPUT_DIR = Path(r"SHAP_Analysis/xai_selected_images")

OUTPUT_DIR = Path(r"SHAP_Analysis/xai_shap_results")
OUTPUT_CSV = OUTPUT_DIR / "xai_shap_log.csv"


IMGSZ = 224
DEVICE = 0

MAX_EVALS = 1000
BATCH_SIZE = 8

SKIP_EXISTING = True

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png"
}


def yolo_predict_proba_factory(
    model: YOLO,
    imgsz=224,
    device=0
):
    """
    Wrap YOLO classification model in a SHAP-compatible
    prediction function.

    Input:
        (N, H, W, 3) RGB image arrays

    Output:
        (N, C) class probability matrix
    """

    def predict(images):

        pil_images = [
            Image.fromarray(
                image.astype(np.uint8)
            )
            for image in images
        ]

        predictions = model.predict(
            pil_images,
            imgsz=imgsz,
            device=device,
            verbose=False
        )

        probabilities = []

        for prediction in predictions:

            probabilities.append(
                prediction.probs.data
                .detach()
                .cpu()
                .numpy()
            )

        return np.stack(
            probabilities,
            axis=0
        )

    return predict

def run_shap_on_image(
    model: YOLO,
    predict_fn,
    image_path,
    output_path,
    imgsz=224,
    device=0,
    max_evals=1000,
    batch_size=8
):
    """
    Classify one image and calculate SHAP values.
    """

    image_path = Path(image_path)
    output_path = Path(output_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )


    # Load image
    image_pil = Image.open(
        image_path
    ).convert("RGB")

    image_array = np.array(
        image_pil,
        dtype=np.uint8
    )


    # Normal YOLO classification
    prediction = model.predict(
        source=image_pil,
        imgsz=imgsz,
        device=device,
        verbose=False
    )[0]


    predicted_class_id = int(
        prediction.probs.top1
    )

    confidence = float(
        prediction.probs.top1conf
    )

    predicted_class = prediction.names[
        predicted_class_id
    ]


    # All class probabilities
    probabilities = (
        prediction.probs.data
        .detach()
        .cpu()
        .numpy()
    )


    print()
    print(
        f"Image: {image_path.name}"
    )

    print(
        f"Prediction: {predicted_class}"
    )

    print(
        f"Confidence: {confidence:.4f}"
    )


    # SHAP Image Masker
    masker = shap.maskers.Image(
        "inpaint_telea",
        image_array.shape
    )


    # SHAP Partition Explainer
    explainer = shap.Explainer(
        predict_fn,
        masker,
        algorithm="partition",
        output_names=[
            prediction.names[i]
            for i in range(
                len(prediction.names)
            )
        ]
    )


    # Calculate SHAP values
    print(
        "Calculating SHAP values..."
    )


    shap_values = explainer(
        image_array[None, ...],
        max_evals=max_evals,
        batch_size=batch_size
    )


    # Create output folder
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    # Save SHAP visualization
    shap.image_plot(
        shap_values,
        image_array[None, ...],
        show=False
    )

    fig = plt.gcf()


    # Probability text for all classes
    class_probabilities = [
        f"{prediction.names[class_id]}: {probability:.4f}"
        for class_id, probability in enumerate(probabilities)
    ]

    probability_line_1 = " | ".join(
        class_probabilities[:3]
    )

    probability_line_2 = " | ".join(
        class_probabilities[3:]
    )


    # Figure title
    fig.suptitle(
        f"Prediction: {predicted_class} "
        f"| Confidence: {confidence:.4f}\n"
        f"{probability_line_1}\n"
        f"{probability_line_2}",
        fontsize=11,
        y=1.08
    )


    plt.savefig(
        output_path,
        bbox_inches="tight",
        dpi=200
    )

    plt.close()


    print(
        f"SHAP saved: {output_path}"
    )


    # Return prediction information
    result = {
        "filename": image_path.name,
        "input_path": str(image_path),
        "predicted_class_id": predicted_class_id,
        "predicted_class": predicted_class,
        "prediction_confidence": confidence,
        "shap_output": str(output_path),
    }


    # Add probability of each class
    for class_id, probability in enumerate(
        probabilities
    ):

        class_name = prediction.names[
            class_id
        ]

        result[
            f"prob_{class_name}"
        ] = float(probability)


    return result


def main():

    # Check input folder
    if not INPUT_DIR.exists():

        raise FileNotFoundError(
            f"Input directory not found: "
            f"{INPUT_DIR}"
        )

    # Create output directory
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # Load YOLO model ONCE
    print(
        f"Loading model:\n{MODEL_PATH}"
    )

    model = YOLO(
        MODEL_PATH
    )

    predict_fn = (
        yolo_predict_proba_factory(
            model=model,
            imgsz=IMGSZ,
            device=DEVICE
        )
    )

    image_paths = sorted([
        path
        for path in INPUT_DIR.rglob("*")
        if (
            path.is_file()
            and
            path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    ])


    print("========================================")

    print("XAI / SHAP ANALYSIS")

    print("========================================")

    print(f"Images found: {len(image_paths)}")


    if len(image_paths) == 0:
        print("No images found.")

        return

    results = []


    for index, image_path in enumerate(
        image_paths,
        start=1
    ):

        print("========================================")

        print(f"Image {index}/{len(image_paths)}")

        print("========================================")


        relative_path = (
            image_path.relative_to(
                INPUT_DIR
            )
        )

        original_class_folder = (
            relative_path.parent
        )

        output_path = (
            OUTPUT_DIR
            / original_class_folder
            / f"{image_path.stem}_shap.png"
        )


        # Skip existing SHAP output
        if (
            SKIP_EXISTING
            and
            output_path.exists()
        ):

            print(f"Already processed, skipping:")

            print(image_path.name)

            continue


        # Extract selected-class folder
        selected_class = (
            image_path.parent.name
        )


        try:

            result = run_shap_on_image(
                model=model,
                predict_fn=predict_fn,
                image_path=image_path,
                output_path=output_path,
                imgsz=IMGSZ,
                device=DEVICE,
                max_evals=MAX_EVALS,
                batch_size=BATCH_SIZE
            )

            result["selected_class"] = selected_class

            result["class_matches_selection"] = (
                selected_class
                ==
                result["predicted_class"]
            )


            results.append(result)


        except Exception as error:

            print()
            print(f"ERROR processing:")

            print(image_path)

            print(f"Reason: {error}")


    # SAVE CSV LOG
    if results:

        results_df = pd.DataFrame(
            results
        )


        # Reorder important columns
        main_columns = [
            "filename",
            "selected_class",
            "predicted_class",
            "prediction_confidence",
            "class_matches_selection",
            "input_path",
            "shap_output",
        ]


        probability_columns = [
            column
            for column in results_df.columns
            if column.startswith("prob_")
        ]


        results_df = results_df[
            main_columns
            +
            probability_columns
        ]


        results_df.to_csv(
            OUTPUT_CSV,
            index=False
        )


        print("========================================")

        print("XAI ANALYSIS COMPLETE")

        print("========================================")

        print(
            f"Images processed: "
            f"{len(results_df)}"
        )

        print("Predictions:")

        print(
            results_df[
                "predicted_class"
            ].value_counts()
            .sort_index()
        )

        print("Class matches original selection:")

        print(
            results_df[
                "class_matches_selection"
            ].value_counts()
        )

        print(f"CSV log saved to:")

        print(OUTPUT_CSV)


        print(f"SHAP results saved to:")

        print(OUTPUT_DIR)


    else:

        print()
        print("No new images were processed.")


if __name__ == "__main__":
    main()