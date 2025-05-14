import cv2
import numpy as np
import os
import logging
import argparse

# Configure logging for the script itself
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

def filter_thin_regions(image_path, output_path, min_thickness_px=3, min_area_after_erosion=5):
    """Filters out regions thinner than min_thickness_px by merging them with neighbors."""
    try:
        logger.info(f"Loading image from: {image_path}")
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            logger.error(f"Image at {image_path} could not be loaded.")
            return False

        processed_img = img_bgr.copy()
        height, width = processed_img.shape[:2]
        unique_colors = np.unique(processed_img.reshape(-1, 3), axis=0)
        logger.info(f"Found {len(unique_colors)} unique colors in the input image.")

        k_erode = max(1, (min_thickness_px -1) // 2)
        erosion_iter_kernel = np.ones((3,3), np.uint8)

        logger.info(f"Minimum thickness: {min_thickness_px}px. Erosion iterations: {k_erode} with 3x3 kernel. Min area after erosion: {min_area_after_erosion}px.")

        colors_processed_count = 0
        for color_val in unique_colors:
            colors_processed_count += 1
            logger.debug(f"Processing color {colors_processed_count}/{len(unique_colors)}: {color_val}")
            
            current_color_mask = cv2.inRange(processed_img, color_val, color_val)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(current_color_mask, connectivity=8)

            for i in range(1, num_labels):
                component_mask_original = (labels == i).astype(np.uint8) * 255
                original_area = stats[i, cv2.CC_STAT_AREA]

                eroded_component_mask = cv2.erode(component_mask_original, erosion_iter_kernel, iterations=k_erode)
                eroded_area = np.sum(eroded_component_mask > 0)

                if eroded_area < min_area_after_erosion:
                    logger.debug(f"Component of color {color_val} (original area {original_area}) is too thin (eroded area {eroded_area}). Attempting to merge.")
                    
                    dilated_kernel = np.ones((3,3), np.uint8)
                    dilated_component_mask = cv2.dilate(component_mask_original, dilated_kernel, iterations=1)
                    neighbor_only_mask = dilated_component_mask & (~component_mask_original)
                    
                    neighbor_pixels_coords = np.where(neighbor_only_mask > 0)
                    if len(neighbor_pixels_coords[0]) == 0:
                        logger.debug(f"Thin component of color {color_val} has no neighbors. Skipping merge.")
                        continue

                    neighbor_colors = processed_img[neighbor_pixels_coords]
                    if len(neighbor_colors) == 0:
                        logger.debug(f"Thin component of color {color_val} - found neighbor mask but no neighbor colors. Skipping.")
                        continue

                    unique_neighbor_colors, counts = np.unique(neighbor_colors, axis=0, return_counts=True)
                    valid_neighbor_indices = [j for j, nc in enumerate(unique_neighbor_colors) if not np.array_equal(nc, color_val)]
                    if not valid_neighbor_indices:
                        logger.debug(f"Thin component of color {color_val} has no *other* color neighbors. Skipping merge.")
                        continue

                    filtered_neighbor_colors = unique_neighbor_colors[valid_neighbor_indices]
                    filtered_counts = counts[valid_neighbor_indices]

                    if len(filtered_neighbor_colors) == 0:
                         logger.debug(f"Thin component of color {color_val} has no *other* color neighbors after filtering. Skipping merge.")
                         continue

                    merge_target_color = filtered_neighbor_colors[np.argmax(filtered_counts)]
                    processed_img[component_mask_original > 0] = merge_target_color
                    logger.info(f"Merged thin component (color {color_val}, area {original_area}) into color {merge_target_color}.")
        
        output_dir_path = os.path.dirname(output_path)
        if not os.path.exists(output_dir_path) and output_dir_path != "":
            os.makedirs(output_dir_path)
            logger.info(f"Created output directory: {output_dir_path}")
        
        cv2.imwrite(output_path, processed_img)
        logger.info(f"Image with thin regions filtered saved to {output_path}")
        return True

    except Exception as e:
        logger.error(f"An error occurred during thin region filtering: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter out thin regions from an image by merging them with neighbors.")
    parser.add_argument("--input_image", required=True, help="Path to the input image (e.g., from apply_good_contour_v4.py).")
    parser.add_argument("--output_image", required=True, help="Path to save the filtered image.")
    parser.add_argument("--min_thickness", type=int, default=3, help="Minimum thickness in pixels for a region to be kept (default: 3).")
    parser.add_argument("--min_area_after_erosion", type=int, default=5, help="Minimum area in pixels a region must have after erosion to be kept (default: 5).")
    args = parser.parse_args()

    logger.info(f"Starting thin region filtering for input: {args.input_image}")

    if not os.path.exists(args.input_image):
        logger.error(f"Error: Input image {args.input_image} not found.")
    else:
        if not filter_thin_regions(args.input_image, args.output_image, 
                                   min_thickness_px=args.min_thickness, 
                                   min_area_after_erosion=args.min_area_after_erosion):
            logger.error("Thin region filtering process failed.")
        else:
            logger.info("Thin region filtering process completed successfully.")

