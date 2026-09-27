from pathlib import Path
import pandas as pd
import geopandas as gpd
import re


# CONFIGURATION
CLASSIFICATION_LOG = Path(
    r"ModelApplication/road_feature_classification_log.csv"
)

INPUT_GPKG = Path(
    r"GSV_Downloader/roads_with_gsv.gpkg"
)

OUTPUT_GPKG = Path(
    r"ModelApplication/HH_road_classified_with_smoothness.gpkg"
)

OUTPUT_LAYER = "classified_roads"

SMOOTHNESS_MAPPING = {
    "0_excellent": "excellent",
    "1_good": "good",
    "2_intermediate": "intermediate",
    "3_bad_paved": "bad",
    "4_bad_unpaved": "bad",

    # 5_unclassified not included
}


def add_smoothness_to_other_tags(other_tags, smoothness_value):
    """
    Add:
        "smoothness"=>"VALUE"
    to the OSM other_tags attribute.
    """

    # Handle empty / NaN tags
    if pd.isna(other_tags) or str(other_tags).strip() == "":
        return f'"smoothness"=>"{smoothness_value}"'

    tags = str(other_tags).strip()

    # Remove existing smoothness tag if one somehow already exists
    tags = re.sub(
        r'(?:^|,)\s*"smoothness"=>"[^"]*"',
        "",
        tags
    )

    # Clean possible leading/trailing commas
    tags = tags.strip().strip(",")

    # Add smoothness
    if tags:
        tags = f'{tags},"smoothness"=>"{smoothness_value}"'
    else:
        tags = f'"smoothness"=>"{smoothness_value}"'

    return tags

df = pd.read_csv(CLASSIFICATION_LOG)

print(f"Classification log entries: {len(df)}")

required_columns = [
    "osmid",
    "predicted_smoothness"
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(
            f"Required column '{col}' not found in classification log.\n"
            f"Available columns: {df.columns.tolist()}"
        )


# Convert osmid to string so that joining is safer
df["osmid"] = df["osmid"].astype(str)

# Convert model class into actual OSM smoothness value
df["smoothness"] = df["predicted_smoothness"].map(
    SMOOTHNESS_MAPPING
)

# Remove unclassified / unmapped results
df = df[df["smoothness"].notna()].copy()

print(
    f"Usable classified road features: {len(df)}"
)


print("\nLoading GeoPackage...")

gdf = gpd.read_file(INPUT_GPKG)

print(f"Input GeoPackage features: {len(gdf)}")
print(f"Available columns: {gdf.columns.tolist()}")


possible_id_columns = [
    "osmid",
    "osm_id",
    "osm_way_id",
    "id"
]

gpkg_id_column = None

for col in possible_id_columns:
    if col in gdf.columns:
        gpkg_id_column = col
        break

if gpkg_id_column is None:
    raise ValueError(
        "Could not find an OSM ID column in the GeoPackage.\n"
        f"Available columns: {gdf.columns.tolist()}"
    )

print(f"Using GeoPackage ID column: {gpkg_id_column}")


# Convert to string for matching
gdf[gpkg_id_column] = (
    gdf[gpkg_id_column]
    .astype(str)
    .str.strip()
)

df["osmid"] = (
    df["osmid"]
    .astype(str)
    .str.strip()
)

classified_osmids = set(df["osmid"])

gdf_classified = gdf[
    gdf[gpkg_id_column].isin(classified_osmids)
].copy()

print(
    f"GeoPackage features matching classification log: "
    f"{len(gdf_classified)}"
)


# JOIN SMOOTHNESS RESULT
smoothness_lookup = (
    df[["osmid", "smoothness"]]
    .drop_duplicates(subset="osmid")
    .set_index("osmid")["smoothness"]
)

gdf_classified["smoothness_temp"] = (
    gdf_classified[gpkg_id_column]
    .map(smoothness_lookup)
)


# REMOVE FEATURES WITHOUT VALID SMOOTHNESS
gdf_classified = gdf_classified[
    gdf_classified["smoothness_temp"].notna()
].copy()

print(
    f"Features with valid smoothness classification: "
    f"{len(gdf_classified)}"
)


# ADD SMOOTHNESS TO other_tags
if "other_tags" not in gdf_classified.columns:
    raise ValueError(
        "'other_tags' column not found in GeoPackage."
    )


gdf_classified["other_tags"] = gdf_classified.apply(
    lambda row: add_smoothness_to_other_tags(
        row["other_tags"],
        row["smoothness_temp"]
    ),
    axis=1
)

gdf_classified = gdf_classified.drop(
    columns=["smoothness_temp"]
)


# SAVE OUTPUT GPKG
if OUTPUT_GPKG.exists():
    OUTPUT_GPKG.unlink()


gdf_classified.to_file(
    OUTPUT_GPKG,
    layer=OUTPUT_LAYER,
    driver="GPKG"
)

print("\n==============================================")
print("FINISHED")
print("==============================================")

print(f"Input GPKG features:")
print(f"  {len(gdf)}")

print(f"\nUsable classifications:")
print(f"  {len(df)}")

print(f"\nFeatures written to output:")
print(f"  {len(gdf_classified)}")

print(f"\nOutput:")
print(f"  {OUTPUT_GPKG}")