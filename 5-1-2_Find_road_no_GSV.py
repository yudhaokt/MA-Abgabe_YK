import time

from pathlib import Path
from datetime import datetime

import requests
import pandas as pd
import geopandas as gpd

from pyproj import Transformer
from shapely.geometry import Point
from tqdm import tqdm


# API CONFIG
GOOGLE_API_KEY = open("src/gmap_api_key.txt", "r").read().strip()


# INPUT
GPKG_PATH = r"src/HH-roads_NO_smoothness.gpkg"
LAYER_NAME = None
OSMID_COLUMN = "osm_id"


# OUTPUT
OUTPUT_DIR = Path("GSV_Downloader")

# Roads WITH Street View
WITH_STREETVIEW_GPKG = (OUTPUT_DIR / "roads_with_gsv.gpkg")
WITH_STREETVIEW_LAYER = "roads_with_gsv"

# Roads WITHOUT Street View
NO_STREETVIEW_GPKG = (OUTPUT_DIR / "roads_no_gsv.gpkg")
NO_STREETVIEW_LAYER = "roads_no_gsv"

# Log file
AVAILABILITY_LOG = (OUTPUT_DIR / "streetview_availability_log.csv")


# CRS SETTINGS
METRIC_CRS = "EPSG:25832"
WGS84_CRS = "EPSG:4326"

WGS84_TO_METRIC = Transformer.from_crs(
    WGS84_CRS,
    METRIC_CRS,
    always_xy=True,
)


# STREET VIEW SEARCH SETTINGS
ROAD_BUFFER_M = 10.0
SEARCH_POINT_SPACING_M = 5.0
GSV_SEARCH_RADIUS_M = 15
GSV_SOURCE = "outdoor"
SLEEP_BETWEEN_METADATA_REQUESTS = 0.02


# PROCESSING SETTINGS
MAX_FEATURES_TO_CHECK = 5
RESUME_FROM_LOG = True


# HELPERS
def prepare_output_dir():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


def read_csv_if_exists(path):

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)

    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def append_log_row(row):

    df = pd.DataFrame([row])

    if AVAILABILITY_LOG.exists():
        df.to_csv(
            AVAILABILITY_LOG,
            mode="a",
            header=False,
            index=False,
        )

    else:
        df.to_csv(
            AVAILABILITY_LOG,
            index=False,
        )


def get_line_parts(geometry):

    if geometry.geom_type == "LineString":
        return [geometry]

    if geometry.geom_type == "MultiLineString":
        return list(geometry.geoms)

    return []


# LOAD ROADS
def load_roads():

    if LAYER_NAME is None:
        roads_original = gpd.read_file(
            GPKG_PATH
        )

    else:
        roads_original = gpd.read_file(
            GPKG_PATH,
            layer=LAYER_NAME,
        )

    if roads_original.crs is None:
        raise ValueError(
            "The input GeoPackage has no CRS."
        )

    if OSMID_COLUMN not in roads_original.columns:
        raise ValueError(
            f"'{OSMID_COLUMN}' was not found. "
            f"Available columns: "
            f"{list(roads_original.columns)}"
        )


    # Remove invalid geometries
    roads_original = roads_original[
        roads_original.geometry.notna()
        & ~roads_original.geometry.is_empty
    ].copy()


    # Keep only line geometries
    roads_original = roads_original[
        roads_original.geometry.geom_type.isin(
            [
                "LineString",
                "MultiLineString",
            ]
        )
    ].copy()


    # Internal unique ID
    roads_original["_check_id"] = (
        roads_original.index.astype(str)
    )


    # Metric copy for calculations
    roads_metric = (
        roads_original
        .to_crs(METRIC_CRS)
        .copy()
    )

    roads_metric["length_m"] = (
        roads_metric.geometry.length
    )


    # Remove very short features
    valid_mask = (
        roads_metric["length_m"] > 1.0
    )

    roads_metric = (
        roads_metric[
            valid_mask
        ].copy()
    )

    roads_original = (
        roads_original.loc[
            roads_metric.index
        ].copy()
    )


    print(
        f"Road features to consider: "
        f"{len(roads_metric)}"
    )

    print(
        "Total road length: "
        f"{roads_metric['length_m'].sum() / 1000:.2f} km"
    )


    return (
        roads_original,
        roads_metric,
    )


# SEARCH POINT GENERATION
def generate_search_points(
    road_geometry,
    spacing_m,
):

    """
    Generate centerline search positions
    in the metric CRS.

    Short line parts use their midpoint.

    Longer parts are sampled from
    spacing/2 and then every spacing metres.
    """

    samples = []


    for part_number, line in enumerate(
        get_line_parts(road_geometry)
    ):
        length = line.length

        if length <= 1.0:
            continue

        if length <= spacing_m:
            distances = [
                length / 2.0
            ]
        else:
            distances = []

            distance = (
                spacing_m / 2.0
            )

            while distance < length:
                distances.append(
                    distance
                )

                distance += spacing_m

        for distance in distances:
            point_metric = (
                line.interpolate(
                    distance
                )
            )

            samples.append(
                {
                    "part_number":
                        part_number,

                    "distance_along_part_m":
                        distance,

                    "point_metric":
                        point_metric,
                }
            )


    return samples


# COORDINATE CONVERSION
def point_metric_to_wgs84(
    point_metric,
):

    point_series = gpd.GeoSeries(
        [point_metric],
        crs=METRIC_CRS,
    ).to_crs(
        WGS84_CRS
    )

    point_wgs = (
        point_series.iloc[0]
    )

    return (
        point_wgs.y,
        point_wgs.x,
    )


# STREET VIEW METADATA
def check_streetview_metadata(
    lat,
    lon,
):

    url = (
        "https://maps.googleapis.com/"
        "maps/api/streetview/metadata"
    )

    params = {

        "location":
            f"{lat},{lon}",

        "radius":
            GSV_SEARCH_RADIUS_M,

        "source":
            GSV_SOURCE,

        "key":
            GOOGLE_API_KEY,
    }


    try:
        response = requests.get(
            url,
            params=params,
            timeout=20,
        )

        response.raise_for_status()
        data = response.json()


    except Exception as exc:
        return {

            "request_ok":
                False,

            "exists":
                False,

            "status":
                None,

            "pano_id":
                None,

            "pano_lat":
                None,

            "pano_lon":
                None,

            "pano_date":
                None,

            "error":
                str(exc),
        }


    status = data.get(
        "status"
    )


    if status == "OK":
        location = data.get(
            "location",
            {},
        )

        return {
            "request_ok":
                True,

            "exists":
                True,

            "status":
                status,

            "pano_id":
                data.get(
                    "pano_id"
                ),

            "pano_lat":
                location.get(
                    "lat"
                ),

            "pano_lon":
                location.get(
                    "lng"
                ),

            "pano_date":
                data.get(
                    "date"
                ),

            "error":
                None,
        }


    # No Street View result
    if status in {
        "ZERO_RESULTS",
        "NOT_FOUND",
    }:
        return {
            "request_ok":
                True,

            "exists":
                False,

            "status":
                status,

            "pano_id":
                None,

            "pano_lat":
                None,

            "pano_lon":
                None,

            "pano_date":
                None,

            "error":
                None,
        }


    # Request problem
    return {
        "request_ok":
            False,

        "exists":
            False,

        "status":
            status,

        "pano_id":
            None,

        "pano_lat":
            None,

        "pano_lon":
            None,

        "pano_date":
            None,

        "error":
            (
                "Metadata API returned "
                f"status: {status}"
            ),
    }


# DISTANCE TO ROAD
def panorama_distance_to_road_m(
    pano_lat,
    pano_lon,
    road_geometry,
):
    x, y = (
        WGS84_TO_METRIC.transform(
            float(pano_lon),
            float(pano_lat),
        )
    )

    pano_point = Point(
        x,
        y,
    )

    return float(
        pano_point.distance(
            road_geometry
        )
    )


# CHECK ONE ROAD FEATURE
def check_road_feature(
    road,
):

    """
    Return one of:

        has_streetview
        no_streetview
        check_incomplete

    The search stops after the first
    accepted panorama is found.
    """

    road_geometry = (
        road.geometry
    )


    search_points = (
        generate_search_points(
            road_geometry,
            SEARCH_POINT_SPACING_M,
        )
    )


    metadata_requests = 0
    panoramas_returned = 0
    panoramas_inside_buffer = 0


    first_pano_id = None
    first_pano_date = None
    first_pano_distance_m = None

    errors = []

    for sample in search_points:
        lat, lon = (
            point_metric_to_wgs84(
                sample["point_metric"]
            )
        )

        meta = (
            check_streetview_metadata(
                lat,
                lon,
            )
        )

        metadata_requests += 1

        if not meta["request_ok"]:
            errors.append(
                meta["error"]
                or str(
                    meta["status"]
                )
            )

            time.sleep(
                SLEEP_BETWEEN_METADATA_REQUESTS
            )

            continue


        if not meta["exists"]:
            time.sleep(
                SLEEP_BETWEEN_METADATA_REQUESTS
            )

            continue

        panoramas_returned += 1

        if (
            meta["pano_lat"] is None
            or
            meta["pano_lon"] is None
        ):
            errors.append(
                "Panorama metadata "
                "contained no coordinates."
            )

            time.sleep(
                SLEEP_BETWEEN_METADATA_REQUESTS
            )

            continue


        distance_m = (
            panorama_distance_to_road_m(
                meta["pano_lat"],
                meta["pano_lon"],
                road_geometry,
            )
        )


        if distance_m <= ROAD_BUFFER_M:
            panoramas_inside_buffer += 1

            first_pano_id = (
                meta["pano_id"]
            )

            first_pano_date = (
                meta["pano_date"]
            )

            first_pano_distance_m = (
                distance_m
            )

            return {
                "status":
                    "has_streetview",

                "metadata_requests":
                    metadata_requests,

                "candidate_locations":
                    len(search_points),

                "panoramas_returned":
                    panoramas_returned,

                "panoramas_inside_buffer":
                    panoramas_inside_buffer,

                "first_pano_id":
                    first_pano_id,

                "first_pano_date":
                    first_pano_date,

                "first_pano_distance_m":
                    first_pano_distance_m,

                "error_count":
                    len(errors),

                "errors":
                    (
                        " | ".join(
                            errors[:5]
                        )
                        if errors
                        else None
                    ),
            }

        time.sleep(
            SLEEP_BETWEEN_METADATA_REQUESTS
        )


    if errors:
        status = (
            "check_incomplete"
        )

    else:
        status = (
            "no_streetview"
        )

    return {
        "status":
            status,

        "metadata_requests":
            metadata_requests,

        "candidate_locations":
            len(search_points),

        "panoramas_returned":
            panoramas_returned,

        "panoramas_inside_buffer":
            panoramas_inside_buffer,

        "first_pano_id":
            first_pano_id,

        "first_pano_date":
            first_pano_date,

        "first_pano_distance_m":
            first_pano_distance_m,

        "error_count":
            len(errors),

        "errors":
            (
                " | ".join(
                    errors[:5]
                )
                if errors
                else None
            ),
    }


# WRITE GPKG
def write_status_gpkg(
    roads_original,
    check_ids,
    output_path,
    layer_name,
    description,
):
    """
    Write selected road features to a GeoPackage.
    """

    result = roads_original[
        roads_original[
            "_check_id"
        ]
        .astype(str)
        .isin(
            set(
                map(
                    str,
                    check_ids,
                )
            )
        )
    ].copy()


    result = result.drop(
        columns=[
            "_check_id"
        ],
        errors="ignore",
    )


    if output_path.exists():
        output_path.unlink()


    if result.empty:
        print(
            f"No {description} "
            "roads were found."
        )

        print(
            "No GeoPackage "
            "was written."
        )

        return


    result.to_file(
        output_path,
        layer=layer_name,
        driver="GPKG",
    )


    print(
        f"Saved {len(result)} "
        f"{description} road feature(s) to:"
    )

    print(
        output_path
    )


# MAIN
def main():
    prepare_output_dir()

    roads_original, roads_metric = (
        load_roads()
    )

    previous_log = (
        read_csv_if_exists(
            AVAILABILITY_LOG
        )
    )


    completed_check_ids = set()
    known_with_streetview_ids = set()
    known_no_streetview_ids = set()


    # READ PREVIOUS RESULTS
    if (
        not previous_log.empty
        and
        "check_id"
        in previous_log.columns
    ):
        if (
            RESUME_FROM_LOG
            and
            "status"
            in previous_log.columns
        ):
            completed = previous_log[
                previous_log[
                    "status"
                ].isin(
                    [
                        "has_streetview",
                        "no_streetview",
                    ]
                )
            ]

            completed_check_ids = set(
                completed[
                    "check_id"
                ]
                .dropna()
                .astype(str)
            )


        # Previously found WITH Street View
        if "status" in previous_log.columns:

            known_with = previous_log[
                previous_log[
                    "status"
                ]
                == "has_streetview"
            ]

            known_with_streetview_ids = set(
                known_with[
                    "check_id"
                ]
                .dropna()
                .astype(str)
            )

            # Previously found WITHOUT Street View
            known_no = previous_log[
                previous_log[
                    "status"
                ]
                == "no_streetview"
            ]

            known_no_streetview_ids = set(
                known_no[
                    "check_id"
                ]
                .dropna()
                .astype(str)
            )


    # SELECT FEATURES TO CHECK
    roads_to_check = roads_metric[
        ~roads_metric[
            "_check_id"
        ]
        .astype(str)
        .isin(
            completed_check_ids
        )
    ].copy()


    if (
        MAX_FEATURES_TO_CHECK
        is not None
    ):
        roads_to_check = (
            roads_to_check.head(
                MAX_FEATURES_TO_CHECK
            )
        )


    print(
        f"Previously completed: "
        f"{len(completed_check_ids)}"
    )

    print(
        f"Features to check "
        f"in this run: "
        f"{len(roads_to_check)}"
    )

    print()


    # NEW RESULTS
    new_with_streetview_ids = set()
    new_no_streetview_ids = set()


    has_count = 0
    no_count = 0
    incomplete_count = 0
    total_metadata_requests = 0


    # PROCESS ROADS
    for road_index, road in tqdm(
        roads_to_check.iterrows(),

        total=len(
            roads_to_check
        ),

        desc=(
            "Checking Street View availability"
        ),
    ):
        result = (
            check_road_feature(
                road
            )
        )

        total_metadata_requests += (
            result[
                "metadata_requests"
            ]
        )

        check_id = str(
            road[
                "_check_id"
            ]
        )

        log_row = {
            "timestamp":
                datetime.now().isoformat(
                    timespec="seconds"
                ),

            "check_id":
                check_id,

            "road_index":
                road_index,

            "osmid":
                road[
                    OSMID_COLUMN
                ],

            "road_length_m":
                road[
                    "length_m"
                ],

            "status":
                result[
                    "status"
                ],

            "candidate_locations":
                result[
                    "candidate_locations"
                ],

            "metadata_requests":
                result[
                    "metadata_requests"
                ],

            "panoramas_returned":
                result[
                    "panoramas_returned"
                ],

            "panoramas_inside_buffer":
                result[
                    "panoramas_inside_buffer"
                ],

            "first_pano_id":
                result[
                    "first_pano_id"
                ],

            "first_pano_date":
                result[
                    "first_pano_date"
                ],

            "first_pano_distance_m":
                result[
                    "first_pano_distance_m"
                ],

            "road_buffer_m":
                ROAD_BUFFER_M,

            "search_spacing_m":
                SEARCH_POINT_SPACING_M,

            "search_radius_m":
                GSV_SEARCH_RADIUS_M,

            "source":
                GSV_SOURCE,

            "error_count":
                result[
                    "error_count"
                ],

            "errors":
                result[
                    "errors"
                ],
        }

        append_log_row(
            log_row
        )


        # STORE RESULT IDS
        if (
            result["status"]
            == "has_streetview"
        ):
            has_count += 1

            new_with_streetview_ids.add(
                check_id
            )

        elif (
            result["status"]
            == "no_streetview"
        ):
            no_count += 1

            new_no_streetview_ids.add(
                check_id
            )

        else:
            incomplete_count += 1


    # COMBINE PREVIOUS + CURRENT RESULTS
    all_with_streetview_ids = (
        known_with_streetview_ids
        |
        new_with_streetview_ids
    )


    all_no_streetview_ids = (
        known_no_streetview_ids
        |
        new_no_streetview_ids
    )

    # WRITE ROADS WITH STREET VIEW

    write_status_gpkg(
        roads_original,
        all_with_streetview_ids,
        WITH_STREETVIEW_GPKG,
        WITH_STREETVIEW_LAYER,
        "with Street View",
    )


    # WRITE ROADS WITHOUT STREET VIEW
    write_status_gpkg(
        roads_original,
        all_no_streetview_ids,
        NO_STREETVIEW_GPKG,
        NO_STREETVIEW_LAYER,
        "without Street View",
    )


    # SUMMARY
    print()

    print("Finished.")

    print(
        f"Has Street View "
        f"(this run): "
        f"{has_count}"
    )

    print(
        f"No Street View "
        f"(this run): "
        f"{no_count}"
    )

    print(
        f"Incomplete checks "
        f"(this run): "
        f"{incomplete_count}"
    )

    print(
        f"Metadata requests made "
        f"(this run): "
        f"{total_metadata_requests}"
    )

    print()

    print(
        "Total confirmed "
        "WITH Street View: "
        f"{len(all_with_streetview_ids)}"
    )

    print(
        "Total confirmed "
        "WITHOUT Street View: "
        f"{len(all_no_streetview_ids)}"
    )

    print()

    print(
        f"With Street View GPKG: "
        f"{WITH_STREETVIEW_GPKG}"
    )

    print(
        f"No Street View GPKG: "
        f"{NO_STREETVIEW_GPKG}"
    )

    print(
        f"Availability log: "
        f"{AVAILABILITY_LOG}"
    )


if __name__ == "__main__":
    main()