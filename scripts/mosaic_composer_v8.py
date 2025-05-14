import cv2
import numpy as np
import os
import logging
import argparse
from shapely.geometry import Polygon
from shapely.validation import make_valid

# --- Configuration ---
MIN_CONTOUR_AREA_DEFAULT = 100 # Default minimum area for a color zone to be processed
CONTOUR_APPROX_EPSILON_FACTOR = 0.005 # Factor for cv2.approxPolyDP

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

def create_cv_contour_list(cv_contour_points):
    """Converts OpenCV contour points to a list of tuples."""
    return [tuple(p[0]) for p in cv_contour_points]

def get_color_zone_polygons(image_path: str, min_area_threshold: int) -> dict[tuple, list[Polygon]]:
    """
    Extracts Shapely Polygons for each unique color zone in the image, handling holes correctly.

    Args:
        image_path: Path to the input image.
        min_area_threshold: Minimum area for a contour to be considered.

    Returns:
        A dictionary where keys are color (R, G, B) tuples and
        values are lists of Shapely Polygons for that color.
    """
    logger.info(f"Loading image for color zone polygon extraction: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Failed to load image: {image_path}")
        return {}

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    unique_colors = np.unique(img_rgb.reshape(-1, 3), axis=0)
    logger.info(f"Found {len(unique_colors)} unique colors in the input image.")

    all_color_polygons = {}

    for color_rgb in unique_colors:
        if tuple(color_rgb) == (255, 255, 255) or tuple(color_rgb) == (0,0,0): # Skip pure white or black if they are background
            # logger.debug(f"Skipping common background color: {color_rgb}")
            pass # For now, let's process all colors, can be adjusted

        lower_bound = np.array(color_rgb, dtype=np.uint8)
        upper_bound = np.array(color_rgb, dtype=np.uint8)
        mask = cv2.inRange(img_rgb, lower_bound, upper_bound)

        # Find all contours and their hierarchy
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        polygons_for_current_color = []
        if hierarchy is not None and len(hierarchy) > 0:
            hierarchy = hierarchy[0] # Get the actual hierarchy array
            for i, contour_points in enumerate(contours):
                # Process only external contours (those with no parent)
                if hierarchy[i][3] == -1:
                    # Approximate contour
                    perimeter = cv2.arcLength(contour_points, True)
                    if perimeter == 0:
                        continue
                    epsilon = CONTOUR_APPROX_EPSILON_FACTOR * perimeter
                    approx_contour_points = cv2.approxPolyDP(contour_points, epsilon, True)

                    if len(approx_contour_points) < 3: # A polygon needs at least 3 vertices
                        continue
                        
                    shell_coords = create_cv_contour_list(approx_contour_points)
                    hole_coords_list = []

                    # Find all holes (children of the current external contour)
                    child_index = hierarchy[i][2] # First child
                    while child_index != -1:
                        # Approximate child contour
                        child_perimeter = cv2.arcLength(contours[child_index], True)
                        if child_perimeter > 0:
                            child_epsilon = CONTOUR_APPROX_EPSILON_FACTOR * child_perimeter
                            approx_child_points = cv2.approxPolyDP(contours[child_index], child_epsilon, True)
                            if len(approx_child_points) >= 3:
                                hole_coords_list.append(create_cv_contour_list(approx_child_points))
                        child_index = hierarchy[child_index][0] # Next sibling

                    try:
                        poly = Polygon(shell=shell_coords, holes=hole_coords_list if hole_coords_list else None)
                        poly = make_valid(poly) # Ensure polygon is valid

                        if poly.area >= min_area_threshold and not poly.is_empty:
                            if poly.geom_type == 'Polygon':
                                polygons_for_current_color.append(poly)
                            elif poly.geom_type == 'MultiPolygon':
                                for p_geom in poly.geoms: # list(poly) in older Shapely, poly.geoms in newer
                                    if p_geom.area >= min_area_threshold and not p_geom.is_empty and p_geom.geom_type == 'Polygon':
                                        polygons_for_current_color.append(p_geom)
                        # else:
                            # logger.debug(f"Polygon for color {color_rgb} rejected. Area: {poly.area}, Min: {min_area_threshold}")

                    except Exception as e:
                        logger.warning(f"Could not create Shapely polygon for color {color_rgb} from CV contour {i}: {e}")

        if polygons_for_current_color:
            all_color_polygons[tuple(color_rgb.tolist())] = polygons_for_current_color
            logger.info(f"Found {len(polygons_for_current_color)} valid polygons for color {color_rgb} (Area >= {min_area_threshold}).")

    return all_color_polygons

def main():
    parser = argparse.ArgumentParser(description="Extracts color zone polygons from an image for mosaic generation.")
    parser.add_argument("--input_image", required=True, help="Path to the input image (e.g., quantized image).")
    # output_dir argument is not used yet, but good to have for future steps
    parser.add_argument("--output_dir", default="./output_mosaic_v8", help="Directory to save results (not used in this initial script).")
    parser.add_argument("--min_contour_area", type=int, default=MIN_CONTOUR_AREA_DEFAULT,
                        help=f"Minimum area of a color zone to process (default: {MIN_CONTOUR_AREA_DEFAULT}).")
    args = parser.parse_args()

    logger.info(f"Starting color zone polygon extraction for: {args.input_image}")

    if not os.path.exists(args.input_image):
        logger.error(f"Error: Input image {args.input_image} not found.")
        return

    output_dir = args.output_dir
    if not os.path.exists(output_dir) and output_dir != "":
        try:
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
        except OSError as e:
            logger.error(f"Could not create output directory {output_dir}: {e}")
            # Decide if we should exit or continue without saving capabilities
            # For now, this script doesn't save, so we can continue.


    color_polygons_map = get_color_zone_polygons(args.input_image, args.min_contour_area)

    if not color_polygons_map:
        logger.warning("No color zone polygons were extracted. Check input image and parameters.")
        return

    total_polygons = 0
    for color, polygons in color_polygons_map.items():
        logger.info(f"Color {color}: Found {len(polygons)} polygons.")
        total_polygons += len(polygons)
        for i, poly in enumerate(polygons):
            logger.debug(f"  Polygon {i+1}: Area = {poly.area:.2f}, IsValid = {poly.is_valid}, Type = {poly.geom_type}")
            logger.debug(f"    Exterior: {len(poly.exterior.coords)} points. Holes: {len(poly.interiors)}")


    logger.info(f"Extraction complete. Total polygons found: {total_polygons}")
    logger.info("Next steps would involve laying tesserae onto these polygons.")

if __name__ == "__main__":
    main() 