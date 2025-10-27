from datetime import datetime, timedelta

import requests
import pandas as pd
import os
import zipfile
import logging
import polyline
import geopy.distance
import concurrent.futures
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Generate GTFS dataset from PMPML API',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog='''
Examples:
  python gtfs_parallel.py                    # Normal execution with validation
  python gtfs_parallel.py --skip-validation  # Skip shape validation steps
  python gtfs_parallel.py --workers 20       # Use 20 parallel workers
  python gtfs_parallel.py --output custom_gtfs --speed 25  # Custom output and speed
    '''
)
parser.add_argument('--skip-validation', action='store_true',
                    help='Skip shape validation and always use fallback distance calculation method')
parser.add_argument('--workers', type=int, default=10,
                    help='Number of parallel workers for fetching route details (default: 10)')
parser.add_argument('--output', type=str, default='gtfs_pmpml',
                    help='Output directory for GTFS files (default: gtfs_pmpml)')
parser.add_argument('--speed', type=float, default=20,
                    help='Average bus speed in km/h for time calculations (default: 20)')
parser.add_argument('--log-level', type=str, default='DEBUG',
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                    help='Logging level (default: DEBUG)')
parser.add_argument('--max-stop-distance', type=float, default=10.0,
                    help='Maximum reasonable distance (km) between consecutive stops for outlier detection (default: 10.0)')

args = parser.parse_args()

# Setup Logging
logging.basicConfig(level=getattr(logging, args.log_level),
                    handlers=[
                        logging.FileHandler("latest.log"),  # Log to file
                        logging.StreamHandler()  # Log to console
                    ]
                    , format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# Base URL for PMPML API
BASE_URL = "https://prod-pmpml-routesapi.chartr.in"
API_KEY = "test"

# Headers for API requests
HEADERS = {"x-api-key": API_KEY}

# Average Bus Speed in km/h
AVERAGE_BUS_SPEED_KMH = args.speed

# Output directory
OUTPUT_DIR = args.output
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuration
SKIP_VALIDATION = args.skip_validation
MAX_WORKERS = args.workers

logger.info("Starting GTFS dataset generation for PMPML...")
if SKIP_VALIDATION:
    logger.warning("⚠️ Shape validation is DISABLED - using fallback method for all routes")

# Dictionaries to store GTFS data
route_mappings = {}
route_polylines = {}
trips_data = [["route_id", "service_id", "trip_id", "trip_headsign", "direction_id", "shape_id"]]
stop_times_data = [["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence", "timepoint"]]
stops_data = [["stop_id", "stop_name", "stop_lat", "stop_lon"]]
shapes_data = [["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"]]
routes_data = [["route_id", "agency_id", "route_short_name", "route_long_name", "route_type"]]
valid_trip_ids, valid_stop_ids, used_shapes = set(), set(), set()
TEST_ENDS = {}  # {"Balewadi Depot"}
# Distance Calculation
def calculate_segment_distance(points):
    """Calculates total distance along a polyline segment."""
    total_distance = 0.0
    for i in range(len(points) - 1):
        distance = geopy.distance.distance(points[i], points[i + 1]).km
        total_distance += distance
    return total_distance


# Precompute distances for all stops in a route
def find_closest_shape_point_in_sequence(shape_points, stop_lat, stop_lon, prev_idx=None):
    """
    Finds the closest shape point within 200m while ensuring sequence follows a valid path.
    - If prev_idx is provided, selects the closest point **after** prev_idx.
    """
    min_distance = float("inf")
    best_idx = None

    for i, (lat, lon) in enumerate(shape_points):
        distance = geopy.distance.distance((stop_lat, stop_lon), (lat, lon)).meters
        if distance < 900:  # Only consider points within 200m
            if prev_idx is None or i >= prev_idx:  # Ensure sequential order
                if distance < min_distance:
                    min_distance = distance
                    best_idx = i

    return best_idx


def detect_and_remove_outlier_stops(route_id, stops, max_distance_km=10.0):
    """
    Detects and removes outlier stops that are unrealistically far from their neighbors.

    An outlier is detected when:
    1. Distance from previous stop to current stop is > max_distance_km
    2. Distance from current stop to next stop is > max_distance_km
    3. Distance from previous to next stop (skipping current) is reasonable (< max_distance_km)

    This indicates the current stop is likely an error in the route data.

    Args:
        route_id: Route identifier for logging
        stops: List of stop dictionaries with 'lat', 'lon', and 'name'
        max_distance_km: Maximum reasonable distance between consecutive stops (default: 10km)

    Returns:
        Filtered list of stops with outliers removed
    """
    if len(stops) <= 2:
        return stops  # Can't detect outliers with too few stops

    filtered_stops = []
    removed_stops = []

    for i in range(len(stops)):
        # Always keep first and last stop
        if i == 0 or i == len(stops) - 1:
            filtered_stops.append(stops[i])
            continue

        prev_stop = stops[i - 1]
        curr_stop = stops[i]
        next_stop = stops[i + 1]

        # Calculate distances in km
        dist_prev_curr = geopy.distance.distance(
            (prev_stop["lat"], prev_stop["lon"]),
            (curr_stop["lat"], curr_stop["lon"])
        ).km

        dist_curr_next = geopy.distance.distance(
            (curr_stop["lat"], curr_stop["lon"]),
            (next_stop["lat"], next_stop["lon"])
        ).km

        dist_prev_next = geopy.distance.distance(
            (prev_stop["lat"], prev_stop["lon"]),
            (next_stop["lat"], next_stop["lon"])
        ).km

        # Check if current stop is an outlier
        is_outlier = (
            dist_prev_curr > max_distance_km and
            dist_curr_next > max_distance_km and
            dist_prev_next < max_distance_km
        )

        if is_outlier:
            removed_stops.append({
                'stop': curr_stop['name'],
                'dist_from_prev': dist_prev_curr,
                'dist_to_next': dist_curr_next,
                'prev_to_next': dist_prev_next
            })
            logger.warning(
                f"⚠️  Outlier detected in route {route_id}: Stop '{curr_stop['name']}' "
                f"({dist_prev_curr:.2f}km from previous, {dist_curr_next:.2f}km to next) "
                f"Direct distance {prev_stop['name']} -> {next_stop['name']}: {dist_prev_next:.2f}km. "
                f"Removing outlier stop."
            )
        else:
            filtered_stops.append(curr_stop)

    if removed_stops:
        logger.info(f"🧹 Route {route_id}: Removed {len(removed_stops)} outlier stop(s)")
        for removed in removed_stops:
            logger.debug(f"   Removed: {removed['stop']} (distances: {removed['dist_from_prev']:.2f}km, {removed['dist_to_next']:.2f}km)")

    return filtered_stops


def refine_shape_points(polyline_points):
    """
    Adds temporary shape points if the gap between two shape points exceeds 50m.
    Ensures shape point density is high enough (~25m apart) for accurate stop matching.
    """
    refined_points = [polyline_points[0]]  # Start with first shape point

    for i in range(1, len(polyline_points)):
        prev_point = refined_points[-1]
        curr_point = polyline_points[i]
        distance = geopy.distance.distance(prev_point, curr_point).meters

        if distance > 50:
            # Insert temporary shape points every 25m
            num_new_points = int(distance / 25)
            lat_step = (curr_point[0] - prev_point[0]) / (num_new_points + 1)
            lon_step = (curr_point[1] - prev_point[1]) / (num_new_points + 1)

            for j in range(num_new_points):
                new_lat = prev_point[0] + lat_step * (j + 1)
                new_lon = prev_point[1] + lon_step * (j + 1)
                refined_points.append((new_lat, new_lon))

        refined_points.append(curr_point)  # Add the original point

    return refined_points


def precompute_route_distances(route_id, polyline_points, stops):
    """
    Improved method to precompute distances between stops along the polyline.
    - Ensures each stop is matched to exactly one shape point.
    - Uses `matched_stop_indices[current_stop_idx]` instead of appending blindly.
    - Moves to the next stop efficiently.
    """
    polyline_points = refine_shape_points(polyline_points)  # Ensure shape density
    stop_distances = []
    matched_stop_indices = [None] * len(stops)  # Use indexing instead of appending
    stop_group_idx = 0
    stop_groups = []
    current_group = []

    logger.info(f"🔄 Processing route {route_id} with {len(stops)} stops and {len(polyline_points)} refined shape points")

    if not stops or not polyline_points:
        logger.error(f"⚠️ No stops or shape points found for route {route_id}")
        return []

    # **Step 1: Group stops within 100m**
    for i in range(len(stops)):
        if i == 0:
            current_group.append(stops[i])
        else:
            prev_stop, curr_stop = stops[i - 1], stops[i]
            stop_distance = geopy.distance.distance(
                (prev_stop["lat"], prev_stop["lon"]),
                (curr_stop["lat"], curr_stop["lon"])
            ).meters

            if stop_distance <= 100:
                current_group.append(curr_stop)
            else:
                stop_groups.append(current_group)
                current_group = [curr_stop]

    stop_groups.append(current_group)  # Add last group

    last_matched_distance = float("inf")
    best_shape_idx = None
    stop_matched = False
    distance_increase_count = 0
    within_range = False  # Flag to indicate when we should start checking
    current_stop_idx = 0  # Track the index of the stop we are processing

    # **Step 2: Match stop groups to shape points**
    shape_idx = 0  # Track the current shape point
    while stop_group_idx < len(stop_groups) and shape_idx < len(polyline_points):
        stop_group = stop_groups[stop_group_idx]
        closest_stop = stop_group[0]  # Assign the first stop in the group
        stop_lat, stop_lon = closest_stop["lat"], closest_stop["lon"]

        while shape_idx < len(polyline_points):
            shape_lat, shape_lon = polyline_points[shape_idx]
            distance = geopy.distance.distance((shape_lat, shape_lon), (stop_lat, stop_lon)).meters

            logger.debug(f"📍 Checking shape point {shape_idx}: {shape_lat}, {shape_lon} | Stop {closest_stop['name']} ({stop_lat}, {stop_lon}) | Distance: {distance:.2f}m")

            # **Start checking when within 200m of the stop**
            if distance < 200:
                within_range = True

            if within_range:
                # **Check if this shape point is the best match**
                if best_shape_idx is None or distance < last_matched_distance:
                    best_shape_idx = shape_idx
                    last_matched_distance = distance
                    stop_matched = True
                    distance_increase_count = 0  # Reset failures
                    logger.debug(f"✅ Stop {closest_stop['name']} matched to shape point {shape_idx} (Updated)")

                # **If two consecutive points fail to update, stop checking**
                else:
                    distance_increase_count += 1
                    logger.debug(f"🔺 Distance increased ({distance_increase_count}) times for stop {closest_stop['name']}")

                    if distance_increase_count >= 2:
                        # **Use indexing instead of appending to ensure each stop gets only one shape match**
                        matched_stop_indices[current_stop_idx] = best_shape_idx

                        # Assign all stops in the group consecutive shape points
                        for i, stop in enumerate(stop_group):
                            shape_idx_offset = best_shape_idx + i
                            if shape_idx_offset < len(polyline_points):
                                matched_stop_indices[current_stop_idx + i] = shape_idx_offset
                                logger.info(f"🟢 Grouped stop {stop['name']} assigned to shape point {shape_idx_offset}")

                        # **Move to the next stop group**
                        logger.info(f"🔀 Moving to next stop group: {[s['name'] for s in stop_groups[stop_group_idx]]} → Next stop group")
                        stop_group_idx += 1
                        last_matched_distance = float("inf")
                        best_shape_idx = None
                        stop_matched = False
                        within_range = False  # Reset range tracking
                        current_stop_idx += len(stop_group)  # Move to the next stop index
                        break  # Stop checking further shape points for this stop

            shape_idx += 1  # Move to the next shape point

    # **Ensure last stop is saved at the last shape point**
    if best_shape_idx is not None:
        matched_stop_indices[current_stop_idx] = best_shape_idx
        logger.info(f"🟢 Final stop {closest_stop['name']} assigned to shape point {best_shape_idx}")

    # **Fix unmatched stops count calculation**
    unmatched_stops = sum(1 for match in matched_stop_indices if match is None)

    if unmatched_stops > 0:
        logger.error(f"⚠️ Some stops in route {route_id} were not matched! Unmatched stops: {unmatched_stops}")
        logger.error(f"⏳ Falling back to old distance calculation method for route {route_id}")
        return precompute_route_distances_fallback(route_id, polyline_points, stops)

    # Compute distances using matched points
    for i in range(1, len(matched_stop_indices)):
        segment_distance = calculate_segment_distance(
            polyline_points[matched_stop_indices[i - 1]:matched_stop_indices[i] + 1]
        )
        logger.info(f"📏 Distance from stop {stops[i-1]['name']} to {stops[i]['name']}: {segment_distance:.2f} km")
        stop_distances.append(segment_distance)

    return stop_distances


def precompute_route_distances_fallback(route_id, polyline_points, stops):
    """Fallback method for precomputing distances when primary method fails."""
    stop_distances = []
    prev_idx = None

    logger.warning(f"⚠️ Using fallback distance calculation for route {route_id}")

    for i, stop in enumerate(stops):
        curr_idx = find_closest_shape_point_in_sequence(polyline_points, stop["lat"], stop["lon"], prev_idx)
        if curr_idx is None:
            logger.warning(f"⚠️ No valid shape point found for stop {stop['name']} in route {route_id}")
            stop_distances.append(0)
            continue

        if prev_idx is not None:
            segment_distance = calculate_segment_distance(
                polyline_points[prev_idx:curr_idx + 1]
            )
            logger.info(f"📏 [Fallback] Distance from {stops[i-1]['name']} to {stops[i]['name']}: {segment_distance:.2f} km")
            stop_distances.append(segment_distance)

        prev_idx = curr_idx

    return stop_distances



# Fetch Routes
logger.info("Fetching bus routes from PMPML API...")
try:
    response = requests.get(f"{BASE_URL}/routes", headers=HEADERS)
    routes_json = response.json()

    for route in routes_json["routes"]:
        if TEST_ENDS and route['long_name'] not in TEST_ENDS:
            logging.debug(f"Skipping route {route['long_name']}")
            continue
        route_id, route_long_name = route["id"], route["long_name"]
        route_mappings[route_long_name] = route_id

        short_name = route["route"]
        long_name = f"{route['start']} to {route['end']}"
        routes_data.append([route_id, "PMPML", short_name, long_name, 3])  # 3 = Bus

        try:
            decoded_points = polyline.decode(route["polyline"])
            route_polylines[route_id] = decoded_points
            for seq, (lat, lon) in enumerate(decoded_points):
                shapes_data.append([f"shape_{route_id}", lat, lon, seq + 1])
        except Exception as e:
            logger.error(f"Error decoding polyline for route {route_long_name}: {e}")

except requests.exceptions.RequestException as e:
    logger.error(f"Error fetching routes: {e}")


# Calculate GTFS-compatible time (allowing for hours > 24:00)
def format_gtfs_time(cumulative_time, trip_start_time):
    """Formats time for GTFS allowing hours > 24 for trips crossing midnight."""
    trip_start_hour = pd.to_datetime(trip_start_time).hour
    current_hour = cumulative_time.hour
    adjusted_hour = current_hour if current_hour >= trip_start_hour else current_hour + 24
    if adjusted_hour > 24:
        logging.debug(f"Adjusted hour while formatting gtfs time {adjusted_hour}")
    return f"{adjusted_hour:02}:{cumulative_time.minute:02}:{cumulative_time.second:02}"


# Process Trips
def process_route_trips(route_id, route, trip_schedules):
    """Processes all trips for a route using precomputed distances."""
    logging.debug(f"Processing route {route['route']}")
    polyline_points = route_polylines.get(route_id, [])

    # Filter out outlier stops
    filtered_stops = detect_and_remove_outlier_stops(route_id, route["stops"], max_distance_km=args.max_stop_distance)

    # Update route with filtered stops
    route["stops"] = filtered_stops

    # Use fallback method if skip validation is enabled
    if SKIP_VALIDATION:
        polyline_points_refined = refine_shape_points(polyline_points) if polyline_points else []
        stop_distances = precompute_route_distances_fallback(route_id, polyline_points_refined, route["stops"])
    else:
        stop_distances = precompute_route_distances(route_id, polyline_points, route["stops"])

    logging.debug(f"Finished computing stop distances {route['route']}, now applying time data to stops.")

    shape_id = f"shape_{route_id}"
    used_shapes.add(shape_id)

    for idx, trip_start_time in enumerate(trip_schedules):
        logging.debug(f"Processing trip {trip_start_time} route {route['route']}")
        trip_id = f"trip_{route_id}_{idx}"
        valid_trip_ids.add(trip_id)

        # route_short_name = route["route"]
        trip_headsign = f"{route['stops'][-1]['name']}"
        trips_data.append([route_id, "WEEKDAY", trip_id, trip_headsign, route["direction"], shape_id])

        cumulative_time = pd.to_datetime(trip_start_time)
        stop_sequence = 1

        for stop, segment_distance in zip(route["stops"], [0] + stop_distances):
            timepoint = 1 if stop_sequence == 1 else 0
            arrival_time = departure_time = format_gtfs_time(cumulative_time, trip_start_time)
            stop_times_data.append([trip_id, arrival_time, departure_time, stop["stop_id"], stop_sequence, timepoint])

            valid_stop_ids.add(stop["stop_id"])
            stops_data.append([stop["stop_id"], stop["name"].title(), stop["lat"], stop["lon"]])

            travel_time_minutes = max((segment_distance / AVERAGE_BUS_SPEED_KMH) * 60, 1)
            if idx == 0:
                logger.debug(f"stop {stop['name']} distance {segment_distance} time {travel_time_minutes} route {route['route']}")
            cumulative_time += pd.Timedelta(minutes=travel_time_minutes)
            stop_sequence += 1
    logger.debug(f"Finished processing route {route['route']}")


# Fetch trip details in parallel
def fetch_route_details(route_long_name, route_id):
    """Fetch transit route details for a given route."""
    try:
        response = requests.get(f"{BASE_URL}/transit_route_details?route={route_long_name}", headers=HEADERS)
        response.raise_for_status()
        route_details_json = response.json()

        for route in route_details_json.get("transit_route", []):
            trip_schedules = route.get("trips_schedule", [])
            if not trip_schedules:
                logger.warning(f"No trip schedules found for route {route_long_name}")
                continue
            process_route_trips(route_id, route, trip_schedules)

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching stop times for route {route_long_name}: {e}")


# Execute API calls in parallel
logger.info(f"Fetching stop times for all routes in parallel using {MAX_WORKERS} workers...")
# [fetch_route_details(route_long_name, route_id) for route_long_name, route_id in route_mappings.items()]
with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(fetch_route_details, route_long_name, route_id) for route_long_name, route_id in route_mappings.items()]
    concurrent.futures.wait(futures)  # Ensure all tasks complete before proceeding
    logging.debug(f"Finished processing all routes.")

# Save GTFS files
# Convert lists to DataFrame
routes_df = pd.DataFrame(routes_data).drop_duplicates()
trips_df = pd.DataFrame(trips_data).drop_duplicates()
stop_times_df = pd.DataFrame(stop_times_data).drop_duplicates()
stops_df = pd.DataFrame(stops_data).drop_duplicates()
shapes_df = pd.DataFrame(shapes_data).drop_duplicates()

routes_df.to_csv(f"{OUTPUT_DIR}/routes.txt", index=False, header=False)
trips_df.to_csv(f"{OUTPUT_DIR}/trips.txt", index=False, header=False)
stop_times_df.to_csv(f"{OUTPUT_DIR}/stop_times.txt", index=False, header=False)
stops_df.to_csv(f"{OUTPUT_DIR}/stops.txt", index=False, header=False)
shapes_df.to_csv(f"{OUTPUT_DIR}/shapes.txt", index=False, header=False)

# Create GTFS-required files
with open(f"{OUTPUT_DIR}/agency.txt", "w") as f:
    f.write(
        "agency_id,agency_name,agency_url,agency_timezone,agency_phone,agency_email\nPMPML,Pune Mahanagar Parivahan Mahamandal Limited,https://www.pmpml.org,Asia/Kolkata,+91 02024545454,complaints@pmpml.org\n")

with open(f"{OUTPUT_DIR}/feed_info.txt", "w") as f:
    f.write("feed_publisher_name,feed_publisher_url,feed_lang,feed_start_date,feed_end_date\nAayush Rai,https://github.com/croyla,en,"
            f"{datetime.now().strftime('%Y%m%d')},{(datetime.now() + timedelta(days=180)).strftime('%Y%m%d')}\n")

with open(f"{OUTPUT_DIR}/calendar.txt", "w") as f:
    f.write(
        "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\nWEEKDAY,1,1,1,1,1,1,1,"
        f"{datetime.now().strftime('%Y%m%d')},{(datetime.now() + timedelta(days=180)).strftime('%Y%m%d')}\n")


# Zip the GTFS dataset with compression
logger.info("Compressing GTFS dataset into ZIP archive...")
gtfs_zip_path = "pmpml_gtfs.zip"
ignored_files = ['.DS_Store']
with zipfile.ZipFile(gtfs_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    for file in os.listdir(OUTPUT_DIR):
        if file in ignored_files:
            continue
        file_path = os.path.join(OUTPUT_DIR, file)
        zipf.write(file_path, arcname=file)
        logger.debug(f"Added {file} to archive")

# Get file sizes for logging
uncompressed_size = sum(os.path.getsize(os.path.join(OUTPUT_DIR, f)) for f in os.listdir(OUTPUT_DIR))
compressed_size = os.path.getsize(gtfs_zip_path)
compression_ratio = (1 - compressed_size / uncompressed_size) * 100 if uncompressed_size > 0 else 0

logger.info(f"✅ GTFS dataset successfully created: {gtfs_zip_path}")
logger.info(f"📦 Uncompressed size: {uncompressed_size / 1024:.2f} KB")
logger.info(f"📦 Compressed size: {compressed_size / 1024:.2f} KB")
logger.info(f"📦 Compression ratio: {compression_ratio:.1f}%")
