import cv2
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def analyze_image_colors(image_path, image_desc):
    """Analyzes an image and prints its unique colors."""
    try:
        logger.info(f"Analyzing colors in: {image_desc} ({image_path})")
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            logger.error(f"Image at {image_path} could not be loaded.")
            return None, False
        
        pixels = img_bgr.reshape(-1, 3)
        unique_colors_bgr = np.unique(pixels, axis=0)
        
        logger.info(f"Found {len(unique_colors_bgr)} unique colors in {os.path.basename(image_path)}:")
        colors_rgb_hex = []
        is_black_present_flag = False
        for color_bgr in unique_colors_bgr:
            color_rgb = color_bgr[::-1] 
            hex_color = "#{:02x}{:02x}{:02x}".format(color_rgb[0], color_rgb[1], color_rgb[2])
            logger.info(f"  BGR: {color_bgr}, RGB: {color_rgb}, HEX: {hex_color}")
            colors_rgb_hex.append({"bgr": color_bgr.tolist(), "rgb": color_rgb.tolist(), "hex": hex_color})
            if all(c == 0 for c in color_rgb):
                is_black_present_flag = True
        
        if is_black_present_flag:
            logger.info(f"Pure black (RGB: [0, 0, 0]) IS present in {image_desc}.")
        else:
            logger.info(f"Pure black (RGB: [0, 0, 0]) IS NOT present in {image_desc}.")
        return colors_rgb_hex, is_black_present_flag

    except Exception as e:
        logger.error(f"An error occurred during color analysis for {image_desc}: {e}")
        import traceback
        traceback.print_exc()
        return None, False

if __name__ == "__main__":
    quantized_image = "/home/ubuntu/processed_mosaic_images/appolo_quantized_8_colors_v3_input.png"
    contoured_image = "/home/ubuntu/processed_mosaic_images/appolo_good_contours_v4.png"
    filtered_image = "/home/ubuntu/processed_mosaic_images/appolo_filtered_thin_v5.png"

    logger.info("--- Starting Diagnostic Color Analysis ---")
    
    palette_quantized, black_in_quantized = analyze_image_colors(quantized_image, "Quantized Image (Baseline Palette)")
    print("-" * 50)
    palette_contoured, black_in_contoured = analyze_image_colors(contoured_image, "Contoured Image (After good_contour_v4.py fix)")
    print("-" * 50)
    palette_filtered, black_in_filtered = analyze_image_colors(filtered_image, "Filtered Image (After filter_thin_lines_v5.py)")
    print("-" * 50)

    logger.info("--- Diagnostic Summary ---")
    if black_in_quantized:
        logger.warning("Black was already in the quantized palette. This is unexpected for this image.")
    else:
        logger.info("Black was NOT in the quantized palette (Correct). ")

    if black_in_contoured:
        logger.error("Black IS PRESENT in the output of apply_good_contour_v4.py. The fix for hole filling was insufficient or there is another source.")
    else:
        logger.info("Black is NOT present in the output of apply_good_contour_v4.py (Fix for hole filling seems to have worked at that stage). ")
    
    if black_in_filtered:
        logger.error("Black IS PRESENT in the final output of filter_thin_lines_v5.py. This script is introducing/merging to black.")
    else:
        logger.info("Black is NOT present in the final output of filter_thin_lines_v5.py (If previous stage was clean, this is good). ")

    if not black_in_contoured and black_in_filtered:
        logger.warning("CONCLUSION: filter_thin_lines_v5.py is the script introducing black pixels.")
    elif black_in_contoured:
        logger.warning("CONCLUSION: apply_good_contour_v4.py is still producing or leaving black pixels.")
    else:
        logger.info("CONCLUSION: No black pixels were introduced by apply_good_contour_v4.py or filter_thin_lines_v5.py according to this analysis (assuming filter_thin_lines_v5.py was run on a clean contoured image).")


