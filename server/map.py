import os
import random
from concurrent.futures import ProcessPoolExecutor

from geopy.distance import geodesic
from PIL import Image
from staticmap import CircleMarker, Line, StaticMap
from tqdm import tqdm

# --- Configuration ---
# Germany bounding box (approximate): lat 47.27–55.06, lon 5.87–15.04
GERMANY_BOUNDS = {
    "min_lat": 47.27,
    "max_lat": 55.06,
    "min_lon": 5.87,
    "max_lon": 15.04,
}
OUTPUT_DIR = "path_previews"


def random_german_coordinate():
    lat = random.uniform(GERMANY_BOUNDS["min_lat"], GERMANY_BOUNDS["max_lat"])
    lon = random.uniform(GERMANY_BOUNDS["min_lon"], GERMANY_BOUNDS["max_lon"])
    return (round(lat, 6), round(lon, 6))


def generate_paths(n_paths=100, coords_per_path=5, max_total_distance_m=500):
    if coords_per_path < 2:
        # A path needs at least two points to have a distance
        avg_segment_dist = 0
    else:
        # Calculate the average distance for each segment of the path
        avg_segment_dist = max_total_distance_m / (coords_per_path - 1)

    paths = []
    for _ in range(n_paths):
        path = []
        # 1. Start with a random coordinate anywhere in Germany
        start_point = random_german_coordinate()
        path.append(start_point)
        current_point = start_point

        # 2. Generate subsequent points based on the previous point
        for _ in range(coords_per_path - 1):
            # Add some randomness to segment length for more natural paths
            # This makes the total distance approximate, not a hard limit.
            segment_distance = random.uniform(avg_segment_dist * 0.7, avg_segment_dist * 1.3)
            
            # Pick a random direction (bearing) from 0 to 360 degrees
            bearing = random.uniform(0, 360)

            # 3. Use geopy to calculate the new point
            # The geopy destination function takes a point and calculates a new one
            # based on distance and bearing.
            destination = geodesic(meters=segment_distance).destination(current_point, bearing)
            
            next_point = (round(destination.latitude, 6), round(destination.longitude, 6))
            path.append(next_point)
            current_point = next_point
            
        paths.append(path)
    return paths

def _create_and_save_image(path_coords, image_path, size=300):
    m = StaticMap(size, size, url_template='http://a.tile.openstreetmap.de/{z}/{x}/{y}.png')
    
    m.add_marker(CircleMarker(path_coords[0][::-1], 'green', 8)) # Start
    m.add_marker(CircleMarker(path_coords[-1][::-1], 'red', 8))  # End
    m.add_line(Line([coord[::-1] for coord in path_coords], 'blue', 3))

    image = m.render()
    image.save(image_path)

def create_and_save_image(args):
    index, path_coords = args
    
    if len(path_coords) < 2:
        return f"Skipped path {index}: not enough coordinates."

    image_path = os.path.join(OUTPUT_DIR, f'path_{index}.png')
    _create_and_save_image(path_coords, image_path)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Saving images to the '{OUTPUT_DIR}' directory.")

    n_images = 100
    paths = generate_paths(n_paths=n_images, coords_per_path=5, max_total_distance_m=500)

    with ProcessPoolExecutor() as executor:
        print(f"\nStarting image generation on {executor._max_workers} CPU cores...")
        results = list(tqdm(executor.map(create_and_save_image, enumerate(paths)), total=n_images))

    print(f"{len(results)} images have been created successfully.")


if __name__ == "__main__":
    main()
