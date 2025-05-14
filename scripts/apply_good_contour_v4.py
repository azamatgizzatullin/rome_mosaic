# apply_good_contour_v4.py
import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.segmentation import slic
from skimage.util import img_as_float
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, LineString
from shapely.ops import unary_union, polygonize
import logging
import os
from PIL import Image
from scipy.interpolate import splprep, splev
import argparse

# Configure logging for the script itself
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# --- Constants and Configuration (can be adjusted via args or kept as defaults) ---
DEFAULT_N_COLORS = 8
MIN_AREA_PX_THRESHOLD = 50
APPROX_EPSILON_FACTOR = 0.003
MORPH_OPEN_ITER_INITIAL = 1
MORPH_CLOSE_ITER_INITIAL = 1
MORPH_OPEN_ITER_FINAL = 1
MORPH_CLOSE_ITER_FINAL = 1
MIN_AREA_DIVISION_FACTOR_FINAL = 1.0
APPLY_SPLINE_SMOOTHING = True
SPLINE_S_FACTOR = None
SPLINE_NUM_POINTS_FACTOR = 2
SMOOTH_KERNEL_SIZE = 3

# --- Edge Detection Helper ---
def detect_edges_canny(image_np_rgb, output_path, threshold1=50, threshold2=150):
    logger.info(f"Detecting edges using Canny with thresholds ({threshold1}, {threshold2}).")
    gray_image = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_image, threshold1, threshold2)
    try:
        pil_img_edges = Image.fromarray(edges)
        pil_img_edges.save(output_path)
        logger.info(f"Canny edge map saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save Canny edge map: {e}")
    return edges

# --- Helper Functions ---
def ensure_polygon(geom):
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, Polygon):
        if not geom.is_valid:
            geom = geom.buffer(0)
        return geom if geom.is_valid and geom.area > 0 else None
    if isinstance(geom, MultiPolygon):
        if not geom.geoms:
            return None
        valid_polygons = []
        for p in geom.geoms:
            if not p.is_valid:
                p = p.buffer(0)
            if p.is_valid and p.area > 0:
                valid_polygons.append(p)
        if not valid_polygons:
            return None
        if len(valid_polygons) == 1:
            return valid_polygons[0]
        return MultiPolygon(valid_polygons) if valid_polygons else None
    return None

def polygon_to_mask(geometry, shape_hw):
    mask = np.zeros(shape_hw, dtype=np.uint8)
    if geometry is None or geometry.is_empty:
        return mask.astype(bool)
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    polygons_to_draw = []
    if isinstance(geometry, Polygon):
        polygons_to_draw = [geometry]
    elif isinstance(geometry, MultiPolygon):
        polygons_to_draw = list(geometry.geoms)
    else:
        logger.warning(f"Unsupported geometry type for polygon_to_mask: {type(geometry)}")
        return mask.astype(bool)

    for poly in polygons_to_draw:
        if poly is None or poly.is_empty:
            continue
        if not poly.is_valid:
            poly = poly.buffer(0)
            if not poly.is_valid or poly.is_empty:
                continue
        exterior = int_coords(poly.exterior.coords)
        if len(exterior) < 3: continue
        cv2.fillPoly(mask, [exterior], 1)
        for interior_coords_shapely in poly.interiors:
            interior = int_coords(interior_coords_shapely.coords)
            if len(interior) < 3: continue
            cv2.fillPoly(mask, [interior], 0)
    return mask.astype(bool)

def smooth_polygon_spline(polygon, num_points_factor=SPLINE_NUM_POINTS_FACTOR, s=SPLINE_S_FACTOR, k=3):
    if not isinstance(polygon, Polygon) or polygon.is_empty:
        return polygon
    smoothed_rings = []
    def smooth_ring(coords_list, is_closed=True):
        if len(coords_list) < k + 1:
            logger.debug(f"Ring has too few points ({len(coords_list)}) for spline smoothing with k={k}. Returning original.")
            return coords_list
        coords_np = np.array(coords_list)
        x = coords_np[:, 0]
        y = coords_np[:, 1]
        if is_closed and not (np.array_equal(coords_np[0], coords_np[-1])):
            x = np.append(x, x[0])
            y = np.append(y, y[0])
        if len(x) < k + 1:
             logger.debug(f"Ring (after closing) has too few points ({len(x)}) for spline smoothing. Returning original.")
             return coords_list
        try:
            tck, u = splprep([x, y], s=s, k=k, per=is_closed)
            num_eval_points = max(int(len(x) * num_points_factor), 50)
            u_new = np.linspace(u.min(), u.max(), num_eval_points)
            x_new, y_new = splev(u_new, tck)
            return list(zip(x_new, y_new))
        except Exception as e:
            logger.error(f"Spline smoothing failed for ring: {e}. Returning original ring.")
            return coords_list

    exterior_coords = list(polygon.exterior.coords)
    smoothed_exterior = smooth_ring(exterior_coords)
    if len(smoothed_exterior) < 3:
        logger.warning("Smoothed exterior has less than 3 points. Returning original.")
        return polygon
    smoothed_interiors = []
    for interior in polygon.interiors:
        interior_coords = list(interior.coords)
        smoothed_interior_ring = smooth_ring(interior_coords)
        if len(smoothed_interior_ring) >= 3:
            smoothed_interiors.append(smoothed_interior_ring)
        else:
            logger.debug("Smoothed interior ring has less than 3 points, discarding.")
    try:
        smoothed_poly = Polygon(smoothed_exterior, smoothed_interiors)
        if not smoothed_poly.is_valid:
            smoothed_poly = smoothed_poly.buffer(0)
        return smoothed_poly if smoothed_poly.is_valid else polygon
    except Exception as e:
        logger.error(f"Failed to create polygon from smoothed rings: {e}. Returning original polygon.")
        return polygon

def preprocess_image(image_np, blur_ksize=(3,3)):
    logger.info("Preprocessing image with Gaussian blur.")
    return cv2.GaussianBlur(image_np, blur_ksize, 0)

def create_color_zones_from_quantized(quantized_image_rgb, original_image_for_edges_rgb, output_canny_map_path,
                                      min_area_px_threshold=MIN_AREA_PX_THRESHOLD,
                                      smooth_kernel_size_val=SMOOTH_KERNEL_SIZE,
                                      approx_epsilon_factor=APPROX_EPSILON_FACTOR,
                                      morph_open_iter_initial=MORPH_OPEN_ITER_INITIAL,
                                      morph_close_iter_initial=MORPH_CLOSE_ITER_INITIAL,
                                      morph_open_iter_final=MORPH_OPEN_ITER_FINAL,
                                      morph_close_iter_final=MORPH_CLOSE_ITER_FINAL,
                                      min_area_division_factor_final=MIN_AREA_DIVISION_FACTOR_FINAL,
                                      apply_spline_smoothing=APPLY_SPLINE_SMOOTHING):
    logger.info("Creating color zones from pre-quantized image.")
    h, w = quantized_image_rgb.shape[:2]
    
    preprocessed_original_for_edges = preprocess_image(original_image_for_edges_rgb)
    edge_map = detect_edges_canny(preprocessed_original_for_edges, output_canny_map_path, threshold1=30, threshold2=100)

    color_zone_polygons = {}
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (smooth_kernel_size_val, smooth_kernel_size_val))
    unique_colors = np.unique(quantized_image_rgb.reshape(-1, 3), axis=0)

    for i, color in enumerate(unique_colors):
        logger.debug(f"Initial processing for color {i+1}/{len(unique_colors)}: {color}")
        mask = cv2.inRange(quantized_image_rgb, color, color)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=morph_open_iter_initial)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=morph_close_iter_initial)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        zone_polygons_for_this_color = []
        for contour in contours:
            if cv2.contourArea(contour) < min_area_px_threshold: continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0: continue
            epsilon = approx_epsilon_factor * perimeter
            approx_contour = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx_contour) < 3: continue
            poly = Polygon([(p[0][0], p[0][1]) for p in approx_contour])
            poly = ensure_polygon(poly)
            if poly and poly.area >= min_area_px_threshold:
                zone_polygons_for_this_color.append(poly)
        if zone_polygons_for_this_color:
            color_tuple = tuple(color.tolist())
            consolidated_geom = unary_union(zone_polygons_for_this_color)
            current_polys_for_color = []
            if isinstance(consolidated_geom, Polygon):
                if consolidated_geom.area >= min_area_px_threshold:
                    current_polys_for_color.append(consolidated_geom)
            elif isinstance(consolidated_geom, MultiPolygon):
                for p_geom in consolidated_geom.geoms:
                    if p_geom.area >= min_area_px_threshold:
                        current_polys_for_color.append(p_geom)
            if current_polys_for_color:
                 color_zone_polygons[color_tuple] = current_polys_for_color

    logger.info(f"Initial pass created {sum(len(polys) for polys in color_zone_polygons.values())} polygon(s) across {len(color_zone_polygons)} colors.")

    final_zones = {}
    processed_overall_mask_shapely = np.zeros((h,w), dtype=bool)
    sorted_colors_tuples = sorted(color_zone_polygons.keys(), key=lambda c: np.mean(c), reverse=True)
    dilated_edges = cv2.dilate(edge_map, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=1)
    edge_barrier_mask_cv = cv2.bitwise_not(dilated_edges)

    for color_tuple in sorted_colors_tuples:
        if color_tuple not in color_zone_polygons: continue
        polygons_for_current_color = color_zone_polygons[color_tuple]
        actual_polygons_for_this_color_final = []
        for poly_candidate in polygons_for_current_color:
            poly_mask_cv = polygon_to_mask(poly_candidate, (h,w)).astype(np.uint8) * 255
            poly_mask_respecting_edges = cv2.bitwise_and(poly_mask_cv, poly_mask_cv, mask=edge_barrier_mask_cv)
            poly_mask_available = np.logical_and(poly_mask_respecting_edges.astype(bool), ~processed_overall_mask_shapely)
            poly_mask_available_uint8 = poly_mask_available.astype(np.uint8) * 255
            opened_mask = cv2.morphologyEx(poly_mask_available_uint8, cv2.MORPH_OPEN, kernel, iterations=morph_open_iter_final)
            closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel, iterations=morph_close_iter_final)
            contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            final_min_area_threshold = min_area_px_threshold / min_area_division_factor_final
            for contour in contours:
                if cv2.contourArea(contour) < final_min_area_threshold: continue
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0: continue
                epsilon = approx_epsilon_factor * perimeter
                approx_contour = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx_contour) < 3: continue
                poly = Polygon([(p[0][0], p[0][1]) for p in approx_contour])
                poly = ensure_polygon(poly)
                if poly and poly.area >= final_min_area_threshold:
                    if apply_spline_smoothing:
                        s_val = SPLINE_S_FACTOR * perimeter if SPLINE_S_FACTOR is not None and perimeter > 0 else None
                        poly_smoothed = smooth_polygon_spline(poly, s=s_val)
                        poly_smoothed = ensure_polygon(poly_smoothed)
                        if poly_smoothed and poly_smoothed.area >= final_min_area_threshold / 1.5:
                            poly = poly_smoothed
                        else:
                            logger.debug(f"Spline smoothing resulted in invalid/small polygon for color {color_tuple}, using non-smoothed.")
                    actual_polygons_for_this_color_final.append(poly)
        if actual_polygons_for_this_color_final:
            final_color_geom = unary_union(actual_polygons_for_this_color_final)
            final_color_geom = ensure_polygon(final_color_geom)
            if final_color_geom:
                final_zones[color_tuple] = final_color_geom
                current_color_drawn_mask = polygon_to_mask(final_color_geom, (h,w))
                processed_overall_mask_shapely = np.logical_or(processed_overall_mask_shapely, current_color_drawn_mask)

    logger.info("Attempting to fill any remaining voids based on original quantized image...")
    remaining_pixels_mask_cv = (~processed_overall_mask_shapely).astype(np.uint8) * 255
    if np.any(remaining_pixels_mask_cv):
        logger.info(f"Found {np.sum(remaining_pixels_mask_cv > 0)} unassigned pixels. Attempting to assign them.")
        colors_in_voids = np.unique(quantized_image_rgb[remaining_pixels_mask_cv > 0], axis=0)
        for color_val_void_np in colors_in_voids:
            color_tuple_void = tuple(color_val_void_np.tolist())
            current_color_in_void_mask = cv2.bitwise_and(cv2.inRange(quantized_image_rgb, color_val_void_np, color_val_void_np), remaining_pixels_mask_cv)
            contours_void, _ = cv2.findContours(current_color_in_void_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            void_polys_for_color = []
            for contour_v in contours_void:
                if cv2.contourArea(contour_v) < min_area_px_threshold / 4: continue # Smaller threshold for voids
                poly_v = Polygon([(p[0][0], p[0][1]) for p in cv2.approxPolyDP(contour_v, approx_epsilon_factor * cv2.arcLength(contour_v, True), True)])
                poly_v = ensure_polygon(poly_v)
                if poly_v and poly_v.area > 0: void_polys_for_color.append(poly_v)
            if void_polys_for_color:
                void_geom_for_color = unary_union(void_polys_for_color)
                void_geom_for_color = ensure_polygon(void_geom_for_color)
                if void_geom_for_color:
                    if color_tuple_void in final_zones:
                        final_zones[color_tuple_void] = unary_union([final_zones[color_tuple_void], void_geom_for_color])
                        final_zones[color_tuple_void] = ensure_polygon(final_zones[color_tuple_void])
                    else:
                        final_zones[color_tuple_void] = void_geom_for_color
    return final_zones

def main():
    parser = argparse.ArgumentParser(description="Apply good contouring to a quantized image based on original image edges.")
    parser.add_argument("--input_quantized_image", required=True, help="Path to the input quantized image (e.g., from quantize_color_v3.py).")
    parser.add_argument("--original_image", required=True, help="Path to the original, non-quantized image (for edge detection).")
    parser.add_argument("--output_contour_image", required=True, help="Path to save the final contoured image.")
    parser.add_argument("--output_canny_map", required=True, help="Path to save the Canny edge map artifact.")
    # Add other parameters from constants if they need to be CLI configurable
    parser.add_argument("--min_area_px", type=int, default=MIN_AREA_PX_THRESHOLD, help="Minimum area for a color zone.")

    args = parser.parse_args()

    logger.info(f"Starting good contour application for quantized image: {args.input_quantized_image}")

    if not os.path.exists(args.input_quantized_image):
        logger.error(f"Input quantized image not found: {args.input_quantized_image}")
        return
    if not os.path.exists(args.original_image):
        logger.error(f"Original image not found: {args.original_image}")
        return

    # Create output directory if it doesn_t exist for the main output image
    output_dir_main = os.path.dirname(args.output_contour_image)
    if not os.path.exists(output_dir_main) and output_dir_main != "":
        os.makedirs(output_dir_main)
        logger.info(f"Created output directory: {output_dir_main}")
    # Create output directory for canny map if different and doesn_t exist
    output_dir_canny = os.path.dirname(args.output_canny_map)
    if not os.path.exists(output_dir_canny) and output_dir_canny != "":
        os.makedirs(output_dir_canny)
        logger.info(f"Created output directory: {output_dir_canny}")

    try:
        quantized_img_bgr = cv2.imread(args.input_quantized_image)
        original_img_bgr = cv2.imread(args.original_image)

        if quantized_img_bgr is None:
            logger.error(f"Failed to load quantized image: {args.input_quantized_image}")
            return
        if original_img_bgr is None:
            logger.error(f"Failed to load original image: {args.original_image}")
            return

        quantized_img_rgb = cv2.cvtColor(quantized_img_bgr, cv2.COLOR_BGR2RGB)
        original_img_rgb = cv2.cvtColor(original_img_bgr, cv2.COLOR_BGR2RGB)

        # Call the main processing function
        final_color_zones = create_color_zones_from_quantized(
            quantized_img_rgb,
            original_img_rgb,
            args.output_canny_map,
            min_area_px_threshold=args.min_area_px
            # Pass other CLI args here if added
        )

        # Create the output image based on final_color_zones
        h, w = quantized_img_rgb.shape[:2]
        output_image_np = np.zeros((h, w, 3), dtype=np.uint8)

        if not final_color_zones:
            logger.warning("No final color zones were generated. Output image will be black.")
        else:
            logger.info(f"Generated {len(final_color_zones)} final color zones. Drawing output image.")
            for color_val_tuple, geometry in final_color_zones.items():
                color_val_np = np.array(color_val_tuple, dtype=np.uint8)
                geom_mask = polygon_to_mask(geometry, (h,w))
                output_image_np[geom_mask] = color_val_np
        
        output_image_bgr = cv2.cvtColor(output_image_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(args.output_contour_image, output_image_bgr)
        logger.info(f"Final contoured image saved to: {args.output_contour_image}")
        logger.info("Good contour application script completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during good contour application: {e}", exc_info=True)

if __name__ == "__main__":
    main()

