# PMPML GTFS Generator

A Python application that fetches transit data from the PMPML (Pune Mahanagar Parivahan Mahamandal Limited) API and generates a GTFS (General Transit Feed Specification) dataset.

## Overview

This tool connects to the Apli-PMPML (Chartr) API to retrieve bus route information, stops, schedules, and polyline data, then processes this information to create a standards-compliant GTFS feed that can be used by transit applications and services.

## Features

- Fetches all PMPML bus routes and their details
- **Automatic outlier stop detection** - Identifies and removes geographically misplaced stops
- Parallel processing of route data for improved performance
- Advanced stop-to-shape matching algorithm with fallback mechanism
- Polyline refinement for accurate distance calculations
- Automatic GTFS dataset generation with proper formatting
- Compressed ZIP archive output
- Detailed logging to both console and file
- Support for midnight-crossing trips
- Configurable via command-line arguments

## Prerequisites

- Python 3.7+
- pip (Python package manager)

## Installation

1. Clone or download this repository

2. Install required dependencies:

```bash
pip install requests pandas polyline geopy
```

Or if using Poetry:

```bash
poetry install
```

## Usage

### Basic Usage

Run the script to generate the complete GTFS dataset:

```bash
python gtfs_parallel.py
```

The script will:
1. Fetch all routes from the Apli-PMPML (Chartr) API
2. Process route details in parallel (default: 10 concurrent workers)
3. Generate GTFS-compliant text files in the `gtfs_pmpml/` directory
4. Create a compressed `pmpml_gtfs.zip` archive
5. Log all operations to `latest.log` and console

### Command Line Options

The application supports several command line arguments for customization:

```bash
python gtfs_parallel.py [OPTIONS]
```

#### Available Options:

**`--skip-validation`**
- Skip shape validation and always use the fallback distance calculation method
- Useful when the primary validation algorithm has issues with certain routes
- Faster execution but may be less accurate
- Example: `python gtfs_parallel.py --skip-validation`

**`--workers N`**
- Set the number of parallel workers for fetching route details
- Default: 10
- Increase for faster processing (limited by CPU/network)
- Example: `python gtfs_parallel.py --workers 20`

**`--output DIRECTORY`**
- Specify the output directory for GTFS files
- Default: `gtfs_pmpml`
- Example: `python gtfs_parallel.py --output custom_output`

**`--speed SPEED`**
- Set the average bus speed in km/h for time calculations
- Default: 20
- Example: `python gtfs_parallel.py --speed 25`

**`--log-level LEVEL`**
- Set the logging verbosity level
- Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`
- Default: `DEBUG`
- Example: `python gtfs_parallel.py --log-level INFO`

**`--max-stop-distance DISTANCE`**
- Maximum reasonable distance (in km) between consecutive stops for outlier detection
- Default: 10.0
- The system automatically removes stops that are unrealistically far from neighboring stops
- Useful for filtering out erroneous stops in route data
- Example: `python gtfs_parallel.py --max-stop-distance 8.0`

**`-h, --help`**
- Show help message with all available options
- Example: `python gtfs_parallel.py --help`

#### Usage Examples:

```bash
# Normal execution with all defaults
python gtfs_parallel.py

# Skip validation for faster processing
python gtfs_parallel.py --skip-validation

# Use 20 workers with custom speed and output directory
python gtfs_parallel.py --workers 20 --speed 25 --output pmpml_data

# Reduce logging noise
python gtfs_parallel.py --log-level INFO

# Combination of options
python gtfs_parallel.py --skip-validation --workers 15 --speed 22 --log-level WARNING
```

### Advanced Configuration

You can also modify the following constants directly in the script for additional customization:

#### API Configuration

Edit the script to change API settings:

```python
BASE_URL = "https://prod-pmpml-routesapi.chartr.in"
API_KEY = "test"
```

- `BASE_URL`: The base URL for the Apli-PMPML (Chartr) API
- `API_KEY`: API authentication key (default: "test")

#### Testing Specific Routes

To test with specific routes only, modify the `TEST_ENDS` variable:

```python
TEST_ENDS = {"Balewadi Depot", "Swargate"}  # Only process these routes
```

Or leave it empty to process all routes:

```python
TEST_ENDS = {}  # Process all routes
```

## Output

The script generates a complete GTFS dataset with the following files:

### Generated Files

1. **agency.txt** - Transit agency information
2. **routes.txt** - Bus route definitions
3. **trips.txt** - Individual trip information
4. **stop_times.txt** - Arrival and departure times for each stop
5. **stops.txt** - Stop locations and names
6. **shapes.txt** - Geographic paths of routes
7. **calendar.txt** - Service schedule information

### Output Archive

- **pmpml_gtfs.zip** - Compressed archive containing all GTFS files

The script provides compression statistics:
- Uncompressed size
- Compressed size
- Compression ratio

## Algorithm Details

### Outlier Stop Detection

The system automatically detects and removes erroneous stops from route data using a geographic analysis algorithm:

#### Detection Criteria

A stop is flagged as an outlier when ALL of the following conditions are met:
1. Distance from previous stop to current stop > `max_stop_distance` (default: 10km)
2. Distance from current stop to next stop > `max_stop_distance`
3. Direct distance from previous stop to next stop (skipping current) < `max_stop_distance`

This pattern indicates the stop is geographically misplaced in the route sequence.

#### Example

For route 119A, if the sequence is:
- Azad Nagar Charholi → **Tapkir Nagar Rahatni** (12km) → Vitbhatti Charholi

But Azad Nagar Charholi → Vitbhatti Charholi is only 2km, then **Tapkir Nagar Rahatni** is identified as an outlier and removed.

#### Features

- Automatically filters out erroneous stops without manual intervention
- Preserves first and last stops of each route
- Logs all removed outliers with distance details
- Configurable threshold via `--max-stop-distance` flag

### Stop Matching Algorithm

The application uses a sophisticated two-step process for matching stops to route shapes:

1. **Stop Grouping**: Groups stops within 100 meters to handle closely-located stops
2. **Sequential Matching**: Matches stop groups to shape points while maintaining route sequence

### Distance Calculation

- Refines polyline points by adding intermediate points every 25 meters
- Uses the Haversine formula (via geopy) for accurate distance calculations
- Calculates segment distances between consecutive stops

### Fallback Mechanism

If the primary matching algorithm fails to match all stops, the system automatically falls back to a simpler nearest-point algorithm.

## Logging

The application generates detailed logs:

- **Console Output**: Real-time progress and important messages
- **latest.log**: Complete log file with debug information

Log entries include:
- Route processing status
- Stop matching details
- Distance calculations
- API request status
- Error and warning messages

## Error Handling

The script includes comprehensive error handling for:
- API connection failures
- Missing or invalid data
- Polyline decoding errors
- Stop matching failures
- File I/O operations

## API Documentation

Refer to `api-doc.yaml` for complete Swagger/OpenAPI specification of the Apli-PMPML (Chartr) API endpoints used by this application.

## GTFS Specification

The output follows the [GTFS specification](https://gtfs.org/schedule/reference/) maintained by Google and the transit community.

### Route Types

The application uses route type `3` (Bus) for all routes as per GTFS standards:
- 0 - Tram
- 1 - Subway
- 2 - Rail
- **3 - Bus** (used by this application)
- 4 - Ferry

### Time Format

Times follow GTFS conventions:
- Format: HH:MM:SS
- Supports hours > 24 for trips crossing midnight (e.g., 25:30:00)

## Performance

- Parallel processing with configurable thread pool
- Efficient stop-to-shape matching algorithm
- In-memory data processing with minimal disk I/O
- Typical runtime: 5-15 minutes for complete PMPML network (varies by API response time)

## Troubleshooting

### Common Issues

1. **API Connection Errors**
   - Check your internet connection
   - Verify the API_KEY is correct in the script
   - Confirm BASE_URL is accessible

2. **Stop Matching Failures**
   - Check log file for specific routes with issues
   - Use `--skip-validation` flag to bypass the primary algorithm
   - The fallback algorithm will automatically engage for failed routes
   - Review polyline quality for problematic routes
   - Example: `python gtfs_parallel.py --skip-validation`

3. **Memory Issues**
   - Reduce worker count using `--workers` flag
   - Process routes in batches using TEST_ENDS variable in the script
   - Example: `python gtfs_parallel.py --workers 5`

4. **Slow Performance**
   - Increase worker count using `--workers` flag (but not beyond your CPU core count)
   - Check network latency to API server
   - Reduce logging verbosity with `--log-level INFO` or `--log-level WARNING`
   - Example: `python gtfs_parallel.py --workers 20 --log-level INFO`

5. **Too Much Log Output**
   - Use the `--log-level` flag to reduce verbosity
   - Options: `INFO`, `WARNING`, or `ERROR`
   - Example: `python gtfs_parallel.py --log-level WARNING`

## File Structure

```
.
├── gtfs_parallel.py          # Main application script
├── api-doc.yaml    # Swagger API documentation
├── README.md                 # This file
├── latest.log                # Log file (generated)
├── gtfs_pmpml/               # Output directory (generated)
│   ├── agency.txt
│   ├── routes.txt
│   ├── trips.txt
│   ├── stop_times.txt
│   ├── stops.txt
│   ├── shapes.txt
│   └── calendar.txt
└── pmpml_gtfs.zip            # Compressed output (generated)
```

## Contributing

To contribute to this project:
1. Test changes with a small subset using TEST_ENDS
2. Ensure logging captures relevant debug information
3. Validate GTFS output using [GTFS Validator](https://gtfs-validator.mobilitydata.org/)

## License

MIT-0

## Contact

For issues related to:
- **PMPML Transit Service**: Contact PMPML at +91 020 2454 5454 or complaints@pmpml.org
- **This Application or dataset**: Open an issue in the project repository

## Acknowledgments

- [Apli-PMPML Android](https://play.google.com/store/apps/details?id=in.chartr.pmpml&hl=en-US)
- [Apli-PMPML iOS](https://apps.apple.com/in/app/aplipmpml/id6739840715)
- [GTFS Specification](https://gtfs.org/)
- [gtfs-validator](https://gtfs-validator.mobilitydata.org/)
- [bmtc-gtfs](https://github.com/Vonter/bmtc-gtfs)