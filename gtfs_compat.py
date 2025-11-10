from datetime import datetime, timedelta
import pandas as pd
import os
import zipfile
import logging
import argparse
from collections import defaultdict

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Transform GTFS dataset by merging UP/DOWN routes',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog='''
Examples:
  python gtfs_compat.py                              # Use default input/output
  python gtfs_compat.py --input pmpml_gtfs.zip       # Specify input ZIP
  python gtfs_compat.py --output gtfs_compat          # Custom output directory
    '''
)
parser.add_argument('--input', type=str, default='pmpml_gtfs.zip',
                    help='Input GTFS ZIP file (default: pmpml_gtfs.zip)')
parser.add_argument('--output', type=str, default='tmp/gtfs',
                    help='Output directory for transformed GTFS files (default: tmp/gtfs)')
parser.add_argument('--log-level', type=str, default='INFO',
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                    help='Logging level (default: INFO)')

args = parser.parse_args()

# Setup Logging
logging.basicConfig(
    level=getattr(logging, args.log_level),
    handlers=[
        logging.FileHandler("gtfs_user.log"),  # Log to file
        logging.StreamHandler()  # Log to console
    ],
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

# Configuration
INPUT_ZIP = args.input
OUTPUT_DIR = args.output
TEMP_EXTRACT_DIR = f"{OUTPUT_DIR}_temp"

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_EXTRACT_DIR, exist_ok=True)

logger.info("Starting GTFS transformation process...")
logger.info(f"Input: {INPUT_ZIP}")
logger.info(f"Output: {OUTPUT_DIR}")

# Step 1: Extract the input GTFS ZIP
logger.info("Extracting input GTFS ZIP file...")
try:
    with zipfile.ZipFile(INPUT_ZIP, 'r') as zip_ref:
        zip_ref.extractall(TEMP_EXTRACT_DIR)
    logger.info(f"✅ Extracted GTFS files to {TEMP_EXTRACT_DIR}")
except Exception as e:
    logger.error(f"Error extracting ZIP file: {e}")
    exit(1)

# Step 2: Read GTFS files into DataFrames
logger.info("Reading GTFS files...")
try:
    routes_df = pd.read_csv(f"{TEMP_EXTRACT_DIR}/routes.txt")
    trips_df = pd.read_csv(f"{TEMP_EXTRACT_DIR}/trips.txt")
    stop_times_df = pd.read_csv(f"{TEMP_EXTRACT_DIR}/stop_times.txt")
    stops_df = pd.read_csv(f"{TEMP_EXTRACT_DIR}/stops.txt")
    shapes_df = pd.read_csv(f"{TEMP_EXTRACT_DIR}/shapes.txt")

    # Read optional files
    agency_df = pd.read_csv(f"{TEMP_EXTRACT_DIR}/agency.txt") if os.path.exists(f"{TEMP_EXTRACT_DIR}/agency.txt") else None
    calendar_df = pd.read_csv(f"{TEMP_EXTRACT_DIR}/calendar.txt") if os.path.exists(f"{TEMP_EXTRACT_DIR}/calendar.txt") else None
    feed_info_df = pd.read_csv(f"{TEMP_EXTRACT_DIR}/feed_info.txt") if os.path.exists(f"{TEMP_EXTRACT_DIR}/feed_info.txt") else None

    logger.info(f"✅ Read {len(routes_df)} routes, {len(trips_df)} trips, {len(stop_times_df)} stop times")
except Exception as e:
    logger.error(f"Error reading GTFS files: {e}")
    exit(1)

# Step 3: Merge UP/DOWN routes
logger.info("Merging UP/DOWN routes...")

# Group routes by route_short_name
route_groups = defaultdict(list)
for idx, row in routes_df.iterrows():
    route_short_name = row['route_short_name']
    route_groups[route_short_name].append(row)

# Create new routes data
new_routes = []
route_id_mapping = {}  # Maps old route_id -> new route_id

for route_short_name, route_list in route_groups.items():
    if len(route_list) == 0:
        continue

    # Use route_short_name as the new route_id
    new_route_id = route_short_name

    # Extract start and end points from route long names
    # Assuming format: "Start to End (UP)" or "Start to End (DOWN)"
    starts = []
    ends = []

    for route in route_list:
        old_route_id = route['route_id']
        route_id_mapping[old_route_id] = new_route_id

        long_name = route['route_long_name']
        # Parse "Start to End (UP/DOWN)" format
        if ' to ' in long_name:
            parts = long_name.split(' to ')
            start = parts[0].strip()
            # Remove (UP) or (DOWN) from end
            end = parts[1].replace('(UP)', '').replace('(DOWN)', '').strip()

            starts.append(start)
            ends.append(end)

    # Create bidirectional long name: "Start <-> End"
    if starts and ends:
        # Use the first start and end (they should be consistent within a route group)
        start_point = starts[0]
        end_point = ends[0]
        new_long_name = f"{start_point} ⇆ {end_point}"
    else:
        # Fallback to first route's long name
        new_long_name = route_list[0]['route_long_name']

    # Create merged route entry
    new_route = {
        'route_id': new_route_id,
        'agency_id': route_list[0]['agency_id'],
        'route_short_name': route_short_name,
        'route_long_name': new_long_name,
        'route_type': route_list[0]['route_type']
    }
    new_routes.append(new_route)

    logger.debug(f"Merged route {route_short_name}: {[r['route_id'] for r in route_list]} -> {new_route_id}")

new_routes_df = pd.DataFrame(new_routes)
logger.info(f"✅ Merged {len(routes_df)} routes into {len(new_routes_df)} routes")

# Step 4: Update trips with new route_ids
logger.info("Updating trips with new route IDs...")
trips_df['route_id'] = trips_df['route_id'].map(route_id_mapping)

# Verify all route_ids were mapped
unmapped = trips_df[trips_df['route_id'].isna()]
if len(unmapped) > 0:
    logger.warning(f"⚠️  {len(unmapped)} trips have unmapped route_ids")

logger.info(f"✅ Updated {len(trips_df)} trips")

# Step 5: Remove 'shape_' prefix from shape_ids
logger.info("Removing 'shape_' prefix from shape IDs...")

def remove_shape_prefix(shape_id):
    if pd.isna(shape_id):
        return shape_id
    return str(shape_id).replace('shape_', '')

# Update shapes.txt
shapes_df['shape_id'] = shapes_df['shape_id'].apply(remove_shape_prefix)

# Update trips.txt to reference shapes without prefix
trips_df['shape_id'] = trips_df['shape_id'].apply(remove_shape_prefix)

logger.info(f"✅ Updated {len(shapes_df)} shape points and {len(trips_df)} trip shape references")

# Step 6: Save transformed GTFS files
logger.info("Saving transformed GTFS files...")

try:
    new_routes_df.to_csv(f"{OUTPUT_DIR}/routes.txt", index=False)
    trips_df.to_csv(f"{OUTPUT_DIR}/trips.txt", index=False)
    stop_times_df.to_csv(f"{OUTPUT_DIR}/stop_times.txt", index=False)
    stops_df.to_csv(f"{OUTPUT_DIR}/stops.txt", index=False)
    shapes_df.to_csv(f"{OUTPUT_DIR}/shapes.txt", index=False)

    # Copy unchanged files
    if agency_df is not None:
        agency_df.to_csv(f"{OUTPUT_DIR}/agency.txt", index=False)
    if calendar_df is not None:
        calendar_df.to_csv(f"{OUTPUT_DIR}/calendar.txt", index=False)
    if feed_info_df is not None:
        feed_info_df.to_csv(f"{OUTPUT_DIR}/feed_info.txt", index=False)

    logger.info("✅ Saved all transformed GTFS files")
except Exception as e:
    logger.error(f"Error saving GTFS files: {e}")
    exit(1)

# Step 7: Create output ZIP file
logger.info("Creating output ZIP file...")
output_zip_path = "./pmpml_gtfs_compat.zip"
logger.info(os.path.dirname(output_zip_path))
os.makedirs(os.path.dirname(output_zip_path), exist_ok=True)

try:
    ignored_files = ['.DS_Store']
    with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in os.listdir(OUTPUT_DIR):
            if file in ignored_files:
                continue
            file_path = os.path.join(OUTPUT_DIR, file)
            zipf.write(file_path, arcname=file)
            logger.debug(f"Added {file} to archive")

    # Get file sizes for logging
    uncompressed_size = sum(os.path.getsize(os.path.join(OUTPUT_DIR, f)) for f in os.listdir(OUTPUT_DIR) if f not in ignored_files)
    compressed_size = os.path.getsize(output_zip_path)
    compression_ratio = (1 - compressed_size / uncompressed_size) * 100 if uncompressed_size > 0 else 0

    logger.info(f"✅ Created output ZIP: {output_zip_path}")
    logger.info(f"📦 Uncompressed size: {uncompressed_size / 1024:.2f} KB")
    logger.info(f"📦 Compressed size: {compressed_size / 1024:.2f} KB")
    logger.info(f"📦 Compression ratio: {compression_ratio:.1f}%")
except Exception as e:
    logger.error(f"Error creating ZIP file: {e}")
    exit(1)

# Step 8: Cleanup temporary directory
logger.info("Cleaning up temporary files...")
try:
    import shutil
    shutil.rmtree(TEMP_EXTRACT_DIR)
    logger.info(f"✅ Removed temporary directory: {TEMP_EXTRACT_DIR}")
except Exception as e:
    logger.warning(f"⚠️  Could not remove temporary directory: {e}")

# Summary
logger.info("\n" + "="*60)
logger.info("TRANSFORMATION SUMMARY")
logger.info("="*60)
logger.info(f"Routes: {len(routes_df)} -> {len(new_routes_df)} (merged)")
logger.info(f"Trips: {len(trips_df)} (updated route references)")
logger.info(f"Shapes: {len(shapes_df['shape_id'].unique())} unique shape IDs (prefix removed)")
logger.info(f"Output: {output_zip_path}")
logger.info("="*60)
logger.info("✅ GTFS transformation completed successfully!")