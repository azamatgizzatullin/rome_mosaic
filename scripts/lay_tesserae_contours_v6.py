import cv2
import numpy as np
import os
import logging
import argparse
from shapely.geometry import Polygon # Not strictly needed for this script's current functionality but kept for consistency if expanded

# Configure logging for the script itself
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# --- Configuration (can be adjusted via args) ---
DEFAULT_TESSERA_WIDTH_PX = 10
DEFAULT_TESSERA_HEIGHT_PX = 15
CONTOUR_APPROX_EPSILON_FACTOR = 0.005
MIN_CONTOUR_AREA_TESSERAE = 100

def get_color_contours(image_path, min_area_threshold):
    """Extracts contours for each unique color in the image."""
    logger.info(f"Loading image for contour extraction: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Failed to load image: {image_path}")
        return None, None
    
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
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter == 0: continue
                    epsilon = CONTOUR_APPROX_EPSILON_FACTOR * perimeter
                    approx_contour = cv2.approxPolyDP(contour, epsilon, True)
                    valid_contours_for_color.append(approx_contour)
        
        if valid_contours_for_color:
            color_contours_map[tuple(color_rgb.tolist())] = valid_contours_for_color
            logger.debug(f"Found {len(valid_contours_for_color)} contours for color {color_rgb} with area >= {min_area_threshold}")
            
    return img, color_contours_map

def lay_tesserae_along_contour_band(canvas, contour, color_bgr, tessera_width, tessera_height):
    """Rudimentary placement of rectangular tesserae along a contour."""
    if len(contour) < 2:
        return

    for i in range(len(contour)):
        pt1 = contour[i][0]
        pt2 = contour[(i + 1) % len(contour)][0]

        center_x = int((pt1[0] + pt2[0]) / 2)
        center_y = int((pt1[1] + pt2[1]) / 2)

        angle_rad = np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
        angle_deg = np.degrees(angle_rad)

        rect = ((center_x, center_y), (tessera_height, tessera_width), angle_deg)
        box = cv2.boxPoints(rect)
        box = box.astype(np.int32)
        cv2.drawContours(canvas, [box], 0, color_bgr, -1)
        cv2.drawContours(canvas, [box], 0, (0,0,0), 1) # Black outline

def main():
    parser = argparse.ArgumentParser(description="Lay tesserae along contours of color zones in an image.")
    parser.add_argument("--input_image", required=True, help="Path to the input image (e.g., from filter_thin_lines_v5.py).")
    parser.add_argument("--output_image", required=True, help="Path to save the image with tesserae contours.")
    parser.add_argument("--tessera_width", type=int, default=DEFAULT_TESSERA_WIDTH_PX, help=f"Approximate width of a tessera in pixels (default: {DEFAULT_TESSERA_WIDTH_PX}).")
    parser.add_argument("--tessera_height", type=int, default=DEFAULT_TESSERA_HEIGHT_PX, help=f"Approximate height of a tessera in pixels (default: {DEFAULT_TESSERA_HEIGHT_PX}).")
    parser.add_argument("--min_contour_area", type=int, default=MIN_CONTOUR_AREA_TESSERAE, help=f"Minimum area of a color zone to process (default: {MIN_CONTOUR_AREA_TESSERAE}).")
    args = parser.parse_args()

    logger.info(f"Starting tesserae contour laying for input: {args.input_image}")

    if not os.path.exists(args.input_image):
        logger.error(f"Error: Input image {args.input_image} not found.")
        return

    output_dir = os.path.dirname(args.output_image)
    if not os.path.exists(output_dir) and output_dir != "":
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

    try:
        original_image_bgr, color_contours_map = get_color_contours(args.input_image, args.min_contour_area)

        if original_image_bgr is None or not color_contours_map:
            logger.error("Could not load image or find contours. Exiting.")
            return

        h, w = original_image_bgr.shape[:2]
        tesserae_canvas_bgr = np.full((h, w, 3), 255, dtype=np.uint8) # White canvas

        logger.info("Laying tesserae along extracted contours...")
        for color_rgb_tuple, contours in color_contours_map.items():
            color_bgr_list = [color_rgb_tuple[2], color_rgb_tuple[1], color_rgb_tuple[0]]
            color_bgr = tuple(c for c in color_bgr_list)
            
            logger.debug(f"Processing {len(contours)} contours for color BGR: {color_bgr}")
            for contour in contours:
                lay_tesserae_along_contour_band(tesserae_canvas_bgr, contour, color_bgr, 
                                                args.tessera_width, args.tessera_height)

        cv2.imwrite(args.output_image, tesserae_canvas_bgr)
        logger.info(f"Tesserae contour layout image saved to {args.output_image}")
        logger.info("Tesserae contour laying script completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during tesserae contour laying: {e}", exc_info=True)

if __name__ == "__main__":
    main()

