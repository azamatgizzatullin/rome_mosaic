import cv2
import numpy as np
import os
import logging
import argparse
from shapely.geometry import Polygon, MultiPolygon, Point, LineString
from shapely.ops import unary_union # triangulate was imported but not used, removed for now
import random

# Configure logging for the script itself
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# --- Configuration (can be adjusted via args) ---
DEFAULT_TESSERA_WIDTH_PX = 10
DEFAULT_TESSERA_HEIGHT_PX = 15
MIN_TESSERA_AREA_PX_FACTOR = 0.25 # Factor of tessera area (width*height*factor)
CONTOUR_APPROX_EPSILON_FACTOR = 0.005
MIN_CONTOUR_AREA_TESSERAE = 100

# --- Helper Functions ---
def get_color_contours(image_path, min_area_threshold):
    """Extracts contours for each unique color in the image."""
    logger.info(f"Loading image for contour extraction: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Failed to load image: {image_path}")
        return None, None, None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    unique_colors = np.unique(img_rgb.reshape(-1, 3), axis=0)
    
    color_contours_map = {}
    logger.info(f"Found {len(unique_colors)} unique colors in the input image.")

    for color_rgb in unique_colors:
        lower_bound = np.array(color_rgb, dtype=np.uint8)
        upper_bound = np.array(color_rgb, dtype=np.uint8)
        mask = cv2.inRange(img_rgb, lower_bound, upper_bound)
        
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_contours_for_color = []
        if hierarchy is not None:
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area >= min_area_threshold:
                    if hierarchy[0][i][3] == -1: # External contour
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter == 0: continue
                        epsilon = CONTOUR_APPROX_EPSILON_FACTOR * perimeter
                        approx_contour = cv2.approxPolyDP(contour, epsilon, True)
                        valid_contours_for_color.append(approx_contour)
        
        if valid_contours_for_color:
            color_contours_map[tuple(color_rgb.tolist())] = valid_contours_for_color
            logger.debug(f"Found {len(valid_contours_for_color)} external contours for color {color_rgb}")
            
    return img, img_rgb, color_contours_map

def create_shapely_polygon(cv_contour):
    if len(cv_contour) < 3:
        return None
    try:
        poly = Polygon([tuple(p[0]) for p in cv_contour])
        if not poly.is_valid:
            poly = poly.buffer(0) # Attempt to fix invalid polygon
        return poly if poly.is_valid else None
    except Exception as e:
        logger.warning(f"Could not create valid Shapely polygon from CV contour: {e}")
        return None

def generate_candidate_tesserae(polygon_to_fill, avg_width, avg_height, min_tessera_area):
    candidates = []
    bounds = polygon_to_fill.bounds
    num_candidates_to_try = 50 # Increased attempts for better coverage

    for _ in range(num_candidates_to_try):
        cx = random.uniform(bounds[0] + avg_width / 2, bounds[2] - avg_width / 2)
        cy = random.uniform(bounds[1] + avg_height / 2, bounds[3] - avg_height / 2)
        
        # For now, axis-aligned rectangles
        half_w, half_h = avg_width / 2, avg_height / 2
        min_x, min_y, max_x, max_y = cx - half_w, cy - half_h, cx + half_w, cy + half_h
        box_poly = Polygon([(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)])
        
        if polygon_to_fill.intersects(box_poly):
            intersection = polygon_to_fill.intersection(box_poly)
            # Ensure intersection is a Polygon or MultiPolygon with significant area
            if isinstance(intersection, (Polygon, MultiPolygon)) and intersection.area > min_tessera_area:
                if isinstance(intersection, Polygon):
                    candidates.append(intersection)
                elif isinstance(intersection, MultiPolygon):
                    for p in intersection.geoms: # Add individual polygons from MultiPolygon
                        if p.area > min_tessera_area:
                            candidates.append(p)
    
    logger.debug(f"Generated {len(candidates)} naive candidate tesserae for a polygon.")
    return candidates

def main():
    parser = argparse.ArgumentParser(description="Fill color zone areas with tesserae in an image.")
    parser.add_argument("--input_image", required=True, help="Path to the input image (e.g., from filter_thin_lines_v5.py or good_contours_v4.py).")
    parser.add_argument("--output_image", required=True, help="Path to save the image with filled tesserae areas.")
    parser.add_argument("--tessera_width", type=int, default=DEFAULT_TESSERA_WIDTH_PX, help=f"Approximate width of a tessera in pixels (default: {DEFAULT_TESSERA_WIDTH_PX}).")
    parser.add_argument("--tessera_height", type=int, default=DEFAULT_TESSERA_HEIGHT_PX, help=f"Approximate height of a tessera in pixels (default: {DEFAULT_TESSERA_HEIGHT_PX}).")
    parser.add_argument("--min_contour_area", type=int, default=MIN_CONTOUR_AREA_TESSERAE, help=f"Minimum area of a color zone to process (default: {MIN_CONTOUR_AREA_TESSERAE}).")
    args = parser.parse_args()

    logger.info(f"Starting tesserae area filling for input: {args.input_image}")

    if not os.path.exists(args.input_image):
        logger.error(f"Error: Input image {args.input_image} not found.")
        return

    output_dir = os.path.dirname(args.output_image)
    if not os.path.exists(output_dir) and output_dir != "":
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

    min_tessera_actual_area = args.tessera_width * args.tessera_height * MIN_TESSERA_AREA_PX_FACTOR

    try:
        original_bgr_img, original_rgb_img, color_contours_map = get_color_contours(args.input_image, args.min_contour_area)

        if original_bgr_img is None or not color_contours_map:
            logger.error("Could not load image or find contours. Exiting.")
            return

        filled_canvas_bgr = original_bgr_img.copy()
        logger.info("Attempting to fill areas within color zones...")

        for color_rgb_tuple, cv_contours_list in color_contours_map.items():
            color_bgr = (color_rgb_tuple[2], color_rgb_tuple[1], color_rgb_tuple[0])
            logger.debug(f"Processing color {color_rgb_tuple} for area filling.")

            for cv_contour in cv_contours_list:
                shapely_poly = create_shapely_polygon(cv_contour)
                if shapely_poly is None or not shapely_poly.is_valid or shapely_poly.area < args.min_contour_area:
                    continue

                logger.debug(f"Attempting to fill polygon with area {shapely_poly.area:.2f}")
                candidate_tesserae_polygons = generate_candidate_tesserae(shapely_poly, 
                                                                        args.tessera_width, 
                                                                        args.tessera_height,
                                                                        min_tessera_actual_area)
                
                for tess_poly in candidate_tesserae_polygons:
                    if tess_poly.is_empty or not isinstance(tess_poly, Polygon):
                        continue
                    exterior_coords = np.array(tess_poly.exterior.coords, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.drawContours(filled_canvas_bgr, [exterior_coords], -1, color_bgr, -1)
                    cv2.drawContours(filled_canvas_bgr, [exterior_coords], -1, (0,0,0), 1) # Black outline

        cv2.imwrite(args.output_image, filled_canvas_bgr)
        logger.info(f"Area filling attempt image saved to {args.output_image}")
        logger.info("Area filling script completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during tesserae area filling: {e}", exc_info=True)

if __name__ == "__main__":
    main()

