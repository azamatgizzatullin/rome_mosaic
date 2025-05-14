import cv2 
import numpy as np
import os
import argparse
import logging

# Configure logging for the script itself
logger = logging.getLogger(__name__)
# Basic config, will be overridden if called by a parent logger with more specific config
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

def quantize_image(image_path, output_path, k=8):
    """Quantizes the colors of an image using k-means clustering."""
    try:
        logger.info(f"Loading image from: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Error: Image at {image_path} could not be loaded.")
            return False

        logger.info("Converting image to RGB.")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pixels = img_rgb.reshape((-1, 3))
        pixels = np.float32(pixels)

        logger.info(f"Performing k-means clustering with k={k}.")
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        quantized_pixels = centers[labels.flatten()]
        quantized_image_rgb = quantized_pixels.reshape(img_rgb.shape)
        quantized_image_bgr = cv2.cvtColor(quantized_image_rgb, cv2.COLOR_RGB2BGR)

        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir) and output_dir != "":
            logger.info(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir)

        cv2.imwrite(output_path, quantized_image_bgr)
        logger.info(f"Quantized image saved to {output_path} with {k} colors.")
        return True
    except Exception as e:
        logger.error(f"An error occurred during quantization: {e}", exc_info=True)
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Quantize colors in an image using k-means clustering.")
    parser.add_argument("--input_image", required=True, help="Path to the input image.")
    parser.add_argument("--output_image", required=True, help="Path to save the quantized image.")
    parser.add_argument("--num_colors", type=int, default=8, help="Number of colors to quantize to (default: 8).")
    args = parser.parse_args()

    logger.info(f"Starting color quantization script for input: {args.input_image}")

    if not os.path.exists(args.input_image):
        logger.error(f"Error: Input image {args.input_image} not found.")
    else:
        if not quantize_image(args.input_image, args.output_image, k=args.num_colors):
            logger.error("Color quantization process failed.")
        else:
            logger.info("Color quantization process completed successfully.")

