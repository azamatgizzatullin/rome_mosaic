# improved_mosaic_generator.py

import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.segmentation import slic # May still be useful for initial broad regions or ideas
from skimage.util import img_as_float # For skimage functions
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, LineString
from shapely.ops import unary_union, polygonize
import logging
import os
from PIL import Image
from scipy.interpolate import splprep, splev

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Constants and Configuration ---
DEFAULT_N_COLORS = 16
DEFAULT_CANVAS_WIDTH_CM = 30.0
DEFAULT_CANVAS_HEIGHT_CM = 30.0
DEFAULT_TESSERA_MIN_SIZE_CM = 0.5
DEFAULT_TESSERA_MAX_SIZE_CM = 5.0
DPI = 100 # Pixels per inch, for cm to px conversion (2.54 cm = 1 inch)

# --- Edge Detection Helper ---
def detect_edges_canny(image_np_rgb, output_dir, threshold1=50, threshold2=150):
    logger.info(f"Detecting edges using Canny with thresholds ({threshold1}, {threshold2}).")
    # image_np_rgb is already preprocessed (e.g., blurred)
    gray_image = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_image, threshold1, threshold2)

    edge_map_path = os.path.join(output_dir, "canny_edge_map.png")
    try:
        pil_img_edges = Image.fromarray(edges)
        pil_img_edges.save(edge_map_path)
        logger.info(f"Canny edge map saved to {edge_map_path}")
    except Exception as e:
        logger.error(f"Failed to save Canny edge map: {e}")
    return edges

# --- Helper Functions ---

def cm_to_px(cm, dpi=DPI):
    return int(cm * dpi / 2.54)

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
        return max(valid_polygons, key=lambda p: p.area) # Return largest valid polygon
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
        cv2.fillPoly(mask, [exterior], 1)
        for interior in poly.interiors:
            interior_coords = int_coords(interior.coords)
            cv2.fillPoly(mask, [interior_coords], 0)
    return mask.astype(bool)


def smooth_polygon_spline(polygon, num_points_factor=2, s=None, k=3):
    """Smooths a Shapely Polygon using B-spline interpolation."""
    if not isinstance(polygon, Polygon) or polygon.is_empty:
        return polygon

    smoothed_rings = []

    def smooth_ring(coords_list, is_closed=True):
        if len(coords_list) < k + 1: # Need at least k+1 points for splprep with degree k
            logger.debug(f"Ring has too few points ({len(coords_list)}) for spline smoothing with k={k}. Returning original.")
            return coords_list
        
        coords_np = np.array(coords_list)
        x = coords_np[:, 0]
        y = coords_np[:, 1]

        # Ensure the ring is closed for splprep if it should be
        if is_closed and not (np.array_equal(coords_np[0], coords_np[-1])):
            x = np.append(x, x[0])
            y = np.append(y, y[0])
        
        if len(x) < k + 1:
             logger.debug(f"Ring (after closing) has too few points ({len(x)}) for spline smoothing with k={k}. Returning original.")
             return coords_list

        try:
            tck, u = splprep([x, y], s=s, k=k, per=is_closed) 
            num_eval_points = max(len(x) * num_points_factor, 50) # Ensure a minimum number of points for smooth curve
            u_new = np.linspace(u.min(), u.max(), num_eval_points)
            x_new, y_new = splev(u_new, tck)
            return list(zip(x_new, y_new))
        except Exception as e:
            logger.error(f"Spline smoothing failed for ring: {e}. Returning original ring.")
            return coords_list

    # Smooth exterior ring
    exterior_coords = list(polygon.exterior.coords)
    smoothed_exterior = smooth_ring(exterior_coords)
    if len(smoothed_exterior) < 3:
        logger.warning("Smoothed exterior has less than 3 points, cannot form polygon. Returning original.")
        return polygon

    # Smooth interior rings
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
            smoothed_poly = smoothed_poly.buffer(0) # Attempt to fix invalid geometry
        return smoothed_poly if smoothed_poly.is_valid else polygon # Fallback to original if smoothing fails
    except Exception as e:
        logger.error(f"Failed to create polygon from smoothed rings: {e}. Returning original polygon.")
        return polygon

# --- Phase 1: Segmentation and Color Zone Preparation ---

def preprocess_image(image_np, blur_ksize=(5,5)):
    logger.info("Preprocessing image with Gaussian blur.")
    return cv2.GaussianBlur(image_np, blur_ksize, 0)

def quantize_image_colors(image_np, n_colors=DEFAULT_N_COLORS):
    logger.info(f"Quantizing image to {n_colors} colors using RGB KMeans.")
    h, w = image_np.shape[:2]
    # No conversion to LAB, use RGB directly
    pixels = image_np.reshape((-1, 3)).astype(np.float32)
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init="auto", max_iter=100)
    labels = kmeans.fit_predict(pixels)
    centers_rgb = kmeans.cluster_centers_.astype(np.uint8) # Centers are directly in RGB
    logger.info(f"KMeans RGB centers: {centers_rgb.tolist()}")
    quantized_rgb_pixels = centers_rgb[labels]
    quantized_rgb_image = quantized_rgb_pixels.reshape((h, w, 3))
    unique_colors_rgb = np.unique(quantized_rgb_image.reshape(-1, 3), axis=0)
    logger.info(f"Actual number of unique colors after RGB quantization: {len(unique_colors_rgb)}")
    return quantized_rgb_image, centers_rgb

def create_color_zones(quantized_image_rgb, edge_map, 
                         min_area_px_threshold=100, 
                         smooth_kernel_size=5, 
                         approx_epsilon_factor=0.005,
                         morph_open_iter_initial=1,
                         morph_close_iter_initial=2,
                         morph_open_iter_final=1,
                         morph_close_iter_final=2,
                         min_area_division_factor_final=2.0,
                         apply_spline_smoothing=True, # New parameter
                         spline_s_factor=None, # Smoothing factor for splprep (None for interpolation)
                         spline_num_points_factor=2):
    logger.info("Creating color zones with refined parameters and optional spline smoothing.")
    h, w = quantized_image_rgb.shape[:2]
    color_zone_polygons = {}
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (smooth_kernel_size, smooth_kernel_size))
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
            epsilon = approx_epsilon_factor * perimeter 
            approx_contour = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx_contour) < 3: continue
            poly = Polygon([(p[0][0], p[0][1]) for p in approx_contour])
            if not poly.is_valid: poly = poly.buffer(0)
            if poly.is_valid and poly.area >= min_area_px_threshold:
                zone_polygons_for_this_color.append(poly)
        if zone_polygons_for_this_color:
            color_tuple = tuple(color.tolist())
            united_color_geom = unary_union(zone_polygons_for_this_color)
            if united_color_geom.is_empty: continue
            current_polys_for_color = []
            if isinstance(united_color_geom, Polygon):
                if united_color_geom.area >= min_area_px_threshold:
                    current_polys_for_color.append(united_color_geom)
            elif isinstance(united_color_geom, (MultiPolygon, GeometryCollection)):
                for p_geom in united_color_geom.geoms:
                    if isinstance(p_geom, Polygon) and p_geom.area >= min_area_px_threshold:
                        current_polys_for_color.append(p_geom)
            if current_polys_for_color:
                 color_zone_polygons[color_tuple] = current_polys_for_color

    logger.info(f"Initial pass created {sum(len(polys) for polys in color_zone_polygons.values())} polygon(s) across {len(color_zone_polygons)} colors.")

    final_zones = {}
    processed_mask = np.zeros((h,w), dtype=bool)
    sorted_colors_tuples = sorted(color_zone_polygons.keys(), key=lambda c: np.mean(c), reverse=True)

    for color_tuple in sorted_colors_tuples:
        color_mask_orig = cv2.inRange(quantized_image_rgb, np.array(color_tuple, dtype=np.uint8), np.array(color_tuple, dtype=np.uint8))
        
        # Combine with edge map to respect contours
        # Dilate edges slightly to make them more robust barriers
        dilated_edges = cv2.dilate(edge_map, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=1)
        # Invert edge map so edges are 0 (barriers) and non-edges are 255 (passable)
        edge_barrier_mask = cv2.bitwise_not(dilated_edges)
        
        # Apply edge barrier to the original color mask
        color_mask_respecting_edges = cv2.bitwise_and(color_mask_orig, color_mask_orig, mask=edge_barrier_mask)

        current_color_mask = np.logical_and(color_mask_respecting_edges.astype(bool), ~processed_mask)
        current_color_mask_uint8 = current_color_mask.astype(np.uint8) * 255
        opened_mask = cv2.morphologyEx(current_color_mask_uint8, cv2.MORPH_OPEN, kernel, iterations=morph_open_iter_final)
        closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel, iterations=morph_close_iter_final)
        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons_for_final_zone = []
        final_min_area_threshold = min_area_px_threshold / min_area_division_factor_final
        for contour in contours:
            if cv2.contourArea(contour) < final_min_area_threshold: continue
            perimeter = cv2.arcLength(contour, True)
            epsilon = approx_epsilon_factor * perimeter 
            approx_contour = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx_contour) < 3: continue
            poly = Polygon([(p[0][0], p[0][1]) for p in approx_contour])
            if not poly.is_valid: poly = poly.buffer(0)
            if poly.is_valid and poly.area >= final_min_area_threshold:
                if apply_spline_smoothing:
                    # Determine s for spline based on perimeter or keep as None for interpolation
                    s_val = spline_s_factor * perimeter if spline_s_factor is not None else None 
                    poly = smooth_polygon_spline(poly, num_points_factor=spline_num_points_factor, s=s_val)
                    if not poly.is_valid: poly = poly.buffer(0)
                    if not poly.is_valid or poly.area < final_min_area_threshold / 2: # Allow some area reduction due to smoothing
                        logger.debug(f"Smoothed polygon invalid or too small, reverting for contour.")
                        # Revert to non-smoothed if smoothing fails badly
                        poly = Polygon([(p[0][0], p[0][1]) for p in approx_contour])
                        if not poly.is_valid: poly = poly.buffer(0)
                        if not poly.is_valid or poly.area < final_min_area_threshold:
                            continue # Skip if even original is bad
                polygons_for_final_zone.append(poly)
        
        if polygons_for_final_zone:
            final_united_geom = unary_union(polygons_for_final_zone)
            if final_united_geom.is_empty: continue
            actual_final_polys = []
            if isinstance(final_united_geom, Polygon):
                if final_united_geom.area >= final_min_area_threshold:
                    actual_final_polys.append(final_united_geom)
            elif isinstance(final_united_geom, (MultiPolygon, GeometryCollection)):
                 for p_geom_final in final_united_geom.geoms:
                    if isinstance(p_geom_final, Polygon) and p_geom_final.area >= final_min_area_threshold:
                        actual_final_polys.append(p_geom_final)
            if actual_final_polys:
                final_zones[color_tuple] = actual_final_polys
                for p_final in actual_final_polys:
                    p_mask = polygon_to_mask(p_final, (h,w))
                    processed_mask = np.logical_or(processed_mask, p_mask)

    logger.info("Attempting to fill any remaining voids to ensure full coverage...")
    remaining_pixels_mask = ~processed_mask
    if np.any(remaining_pixels_mask):
        logger.info(f"Found {np.sum(remaining_pixels_mask)} unassigned pixels. Attempting to assign them.")
        unique_colors_in_voids_rgb = np.unique(quantized_image_rgb[remaining_pixels_mask], axis=0)
        for color_val_void in unique_colors_in_voids_rgb:
            color_tuple_void = tuple(color_val_void.tolist())
            current_color_void_mask = np.logical_and(
                cv2.inRange(quantized_image_rgb, color_val_void, color_val_void).astype(bool),
                remaining_pixels_mask
            )
            current_color_void_mask_uint8 = current_color_void_mask.astype(np.uint8) * 255
            kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            closed_mask_fill = cv2.morphologyEx(current_color_void_mask_uint8, cv2.MORPH_CLOSE, kernel_fill, iterations=1)
            contours_fill, _ = cv2.findContours(closed_mask_fill, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polygons_for_this_void_color = []
            for c_fill in contours_fill:
                if cv2.contourArea(c_fill) < 5: continue # Smallest area for void fills
                perimeter_f = cv2.arcLength(c_fill, True)
                epsilon_f = 0.005 * perimeter_f # Use a small epsilon for void fills
                approx_c_fill = cv2.approxPolyDP(c_fill, epsilon_f, True)
                if len(approx_c_fill) < 3: continue
                poly_f = Polygon([(p[0][0], p[0][1]) for p in approx_c_fill])
                if not poly_f.is_valid: poly_f = poly_f.buffer(0)
                if poly_f.is_valid and poly_f.area > 0:
                    # No spline smoothing for void-filled polygons to keep them simple
                    polygons_for_this_void_color.append(poly_f)
            if polygons_for_this_void_color:
                if color_tuple_void not in final_zones:
                    final_zones[color_tuple_void] = []
                # Add new void-filled polygons to existing ones for that color
                final_zones[color_tuple_void].extend(polygons_for_this_void_color)
                # Update processed_mask with these newly filled areas
                for p_filled in polygons_for_this_void_color:
                    p_mask_filled = polygon_to_mask(p_filled, (h,w))
                    processed_mask = np.logical_or(processed_mask, p_mask_filled)
    
    logger.info(f"Final number of color zones: {sum(len(polys) for polys in final_zones.values())} across {len(final_zones)} colors after void filling.")
    return final_zones

# --- Visualization (NEW FUNCTION - REPLACES OLD ONE) ---
def draw_zones_on_image(image_np_rgb_original, zones_dict, output_path, line_thickness=1):
    logger.info(f"Drawing {sum(len(polys) for polys in zones_dict.values())} color zones on image with fill.")
    # Start with a white canvas of the same size as the original image
    output_image = np.full_like(image_np_rgb_original, (255, 255, 255), dtype=np.uint8)

    for color_tuple, polygons_list in zones_dict.items():
        color_bgr = tuple(reversed(color_tuple)) # OpenCV uses BGR

        for poly_geom in polygons_list: # Iterate through the list of polygons for this color
            if poly_geom is None or poly_geom.is_empty:
                continue

            # Handle both Polygon and MultiPolygon
            geometries_to_draw = []
            if isinstance(poly_geom, Polygon):
                geometries_to_draw.append(poly_geom)
            elif isinstance(poly_geom, MultiPolygon):
                geometries_to_draw.extend(list(poly_geom.geoms))
            else:
                logger.warning(f"Skipping unsupported geometry type {type(poly_geom)} in draw_zones_on_image")
                continue

            for poly in geometries_to_draw:
                if not isinstance(poly, Polygon) or poly.is_empty:
                    continue
                if not poly.is_valid:
                    poly = poly.buffer(0)
                    if not poly.is_valid or poly.is_empty:
                        logger.debug(f"Skipping invalid/empty polygon after buffer(0): {poly.wkt[:100]}")
                        continue

                # Convert Shapely Polygon exterior to OpenCV contour format
                exterior_coords = np.array(poly.exterior.coords, dtype=np.int32).reshape((-1, 1, 2))
                
                # Fill the exterior of the polygon
                cv2.fillPoly(output_image, [exterior_coords], color_bgr)
                
                # Handle interior rings (holes) by filling them with the background color (white)
                for interior in poly.interiors:
                    interior_coords = np.array(interior.coords, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(output_image, [interior_coords], (255, 255, 255)) # Fill holes with white
                
                # Optional: Draw a thin black border around each filled zone for better visual separation if needed
                # cv2.polylines(output_image, [exterior_coords], isClosed=True, color=(0,0,0), thickness=1)
                # for interior in poly.interiors:
                #     interior_coords = np.array(interior.coords, dtype=np.int32).reshape((-1, 1, 2))
                #     cv2.polylines(output_image, [interior_coords], isClosed=True, color=(0,0,0), thickness=1)

    try:
        # Inpainting step to remove white artifacts
        # The output_image is BGR at this point
        inpaint_mask = cv2.inRange(output_image, np.array([255,255,255], dtype=np.uint8), np.array([255,255,255], dtype=np.uint8))
        
        # Check if there's anything to inpaint
        if np.any(inpaint_mask):
            logger.info(f"Found {np.sum(inpaint_mask > 0)} white pixels to inpaint.")
            # Using INPAINT_NS (Navier-Stokes based method)
            output_image_inpainted = cv2.inpaint(output_image, inpaint_mask, 3, cv2.INPAINT_NS) 
        else:
            logger.info("No white artifacts found for inpainting.")
            output_image_inpainted = output_image # No inpainting needed

        # Convert to RGB for PIL
        output_image_rgb = cv2.cvtColor(output_image_inpainted, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(output_image_rgb)
        pil_img.save(output_path)
        logger.info(f"Phase 1 test mosaic with filled zones (and inpainting) saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving output image with inpainting: {e}")


# --- Main Class and Execution ---
class ImprovedMosaicGenerator:
    DEFAULT_PARAMS = {
        "n_colors": DEFAULT_N_COLORS,
        "blur_ksize_w": 5,
        "blur_ksize_h": 5,
        "cz_min_area_px_threshold": 50, # Reduced to allow smaller initial segments
        "cz_smooth_kernel_size": 3,     # Reduced for less aggressive initial smoothing
        "cz_approx_epsilon_factor": 0.002, # Reduced for more detail
        "cz_morph_open_iter_initial": 1,
        "cz_morph_close_iter_initial": 1, # Reduced
        "cz_morph_open_iter_final": 1,
        "cz_morph_close_iter_final": 1,   # Reduced
        "cz_min_area_division_factor_final": 1.5, # Less aggressive reduction
        "apply_spline_smoothing": True,
        "spline_s_factor": 0.05, # e.g., 0.05 for smoothing
        "spline_num_points_factor": 3 # Increased for smoother splines
    }

    def __init__(self, image_path, output_dir="/home/ubuntu/debug_outputs", params=None):
        self.image_path = image_path
        self.output_dir = output_dir
        self.params = self.DEFAULT_PARAMS.copy()
        if params:
            self.params.update(params)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created output directory: {self.output_dir}")
        
        self.image_np_rgb = None
        self.quantized_image_rgb = None
        self.color_centers_rgb = None
        self.final_color_zones = None
        self.edge_map = None # Add edge_map attribute
        self._load_and_preprocess_image()

    def _detect_edges(self):
        if self.image_np_rgb is not None:
            # Use default Canny thresholds or make them configurable in params
            canny_thresh1 = self.params.get("canny_threshold1", 50)
            canny_thresh2 = self.params.get("canny_threshold2", 150)
            self.edge_map = detect_edges_canny(self.image_np_rgb, self.output_dir, threshold1=canny_thresh1, threshold2=canny_thresh2)
            logger.info("Edge detection complete.")
        else:
            logger.error("Cannot detect edges: Image not loaded.")

    def _load_and_preprocess_image(self):
        logger.info(f"Loading image from: {self.image_path}")
        try:
            pil_image = Image.open(self.image_path).convert("RGB")
            self.image_np_rgb = np.array(pil_image)
        except FileNotFoundError:
            logger.error(f"Image file not found at {self.image_path}. Using a dummy image.")
            # Create a dummy image if the original is not found
            dummy_img = Image.new("RGB", (100,100), color="skyblue")
            self.image_np_rgb = np.array(dummy_img)
            # Optionally save the dummy so the path is valid for a re-run if that was the issue
            # dummy_img.save(self.image_path) 
        except Exception as e:
            logger.error(f"Error loading image: {e}. Using a dummy image.")
            dummy_img = Image.new("RGB", (100,100), color="red")
            self.image_np_rgb = np.array(dummy_img)

        if self.image_np_rgb is not None:
            self.image_np_rgb = preprocess_image(self.image_np_rgb, 
                                                 blur_ksize=(self.params["blur_ksize_w"], self.params["blur_ksize_h"]))
            logger.info(f"Image loaded and preprocessed. Shape: {self.image_np_rgb.shape}")
        else:
            logger.error("Failed to load or create a dummy image.")

    def run_phase1_segmentation(self):
        if self.image_np_rgb is None:
            logger.error("Cannot run phase 1: Image not loaded.")
            return

        logger.info("Starting Phase 1: Image Segmentation and Color Zone Preparation")
        
        # Detect edges first
        self._detect_edges()

        self.quantized_image_rgb, self.color_centers_rgb = quantize_image_colors(self.image_np_rgb, self.params["n_colors"])
        
        quantized_output_path = os.path.join(self.output_dir, "quantized_image.png")
        try:
            Image.fromarray(self.quantized_image_rgb).save(quantized_output_path)
            logger.info(f"Quantized image saved to {quantized_output_path}")
        except Exception as e:
            logger.error(f"Failed to save quantized image: {e}")

        self.final_color_zones = create_color_zones(
            self.quantized_image_rgb,
            self.edge_map, # Pass the detected edge map
            min_area_px_threshold=self.params["cz_min_area_px_threshold"],
            smooth_kernel_size=self.params["cz_smooth_kernel_size"],
            approx_epsilon_factor=self.params["cz_approx_epsilon_factor"],
            morph_open_iter_initial=self.params["cz_morph_open_iter_initial"],
            morph_close_iter_initial=self.params["cz_morph_close_iter_initial"],
            morph_open_iter_final=self.params["cz_morph_open_iter_final"],
            morph_close_iter_final=self.params["cz_morph_close_iter_final"],
            min_area_division_factor_final=self.params["cz_min_area_division_factor_final"],
            apply_spline_smoothing=self.params["apply_spline_smoothing"],
            spline_s_factor=self.params["spline_s_factor"],
            spline_num_points_factor=self.params["spline_num_points_factor"]
        )

        if self.final_color_zones:
            logger.info(f"Phase 1 completed. Generated {sum(len(polys) for polys in self.final_color_zones.values())} final color zones.")
            output_image_path = os.path.join(self.output_dir, "improved_mosaic_phase1_output.jpg")
            draw_zones_on_image(self.image_np_rgb, self.final_color_zones, output_image_path)
        else:
            logger.warning("Phase 1 did not generate any color zones.")

    # Placeholder for Phase 2
    def run_phase2_tessellation(self):
        logger.info("Phase 2: Tessellation (Not Implemented)")
        if not self.final_color_zones:
            logger.warning("Cannot run Phase 2: Color zones not generated from Phase 1.")
            return
        # Tessellation logic will go here
        pass

if __name__ == "__main__":
    logger.info("Starting Mosaic Generation Process")
    base_dir = "/home/ubuntu/upload"
    # Ensure the dummy image exists if the real one is missing for testing
    # This part is more for local testing if appolo.jpg might be missing
    # In a real scenario, the image path should be correctly provided.
    image_filename = "appolo.jpg"
    image_full_path = os.path.join(base_dir, image_filename)

    # Check if the intended image exists, if not, create a placeholder in /home/ubuntu/upload
    # This is primarily for robust testing in the sandbox if the file isn't pre-uploaded
    if not os.path.exists(image_full_path):
        logger.warning(f"{image_full_path} not found. Creating a dummy placeholder for testing.")
        # Create a dummy image in the expected upload location if it's missing
        # This helps ensure the script runs even if the image wasn't uploaded in a test scenario
        placeholder_img = Image.new("RGB", (200, 200), color="lightgrey")
        try:
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)
            placeholder_img.save(image_full_path)
            logger.info(f"Created placeholder image at {image_full_path}")
        except Exception as e:
            logger.error(f"Could not create placeholder image: {e}")
            # Fallback to an internal dummy if saving placeholder fails
            image_full_path = "dummy_internal_image.jpg" # This will trigger FileNotFoundError handled by _load_and_preprocess_image

    custom_params = {
        "n_colors": 10, # Trying with fewer colors
        "cz_min_area_px_threshold": 75,
        "cz_smooth_kernel_size": 3,
        "cz_approx_epsilon_factor": 0.0015, # More detail in contours
        "cz_morph_open_iter_initial": 1,
        "cz_morph_close_iter_initial": 1,
        "cz_morph_open_iter_final": 1,
        "cz_morph_close_iter_final": 1,
        "cz_min_area_division_factor_final": 1.2,
        "apply_spline_smoothing": True,
        "spline_s_factor": 0.08, # Increased smoothing for splines
        "spline_num_points_factor": 5 # More points for smoother splines
    }

    generator = ImprovedMosaicGenerator(image_path=image_full_path, params=custom_params)
    generator.run_phase1_segmentation()
    # generator.run_phase2_tessellation() # Uncomment when Phase 2 is ready

    logger.info("Mosaic Generation Process Finished")

