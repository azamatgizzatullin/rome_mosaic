"""
Main script to run the mosaic generation pipeline.
This script orchestrates the execution of individual processing steps.
"""
import argparse
import os
import subprocess
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("MosaicPipeline")

# Define paths to the individual processing scripts
# Assuming main.py is in mosaic_project and scripts are in mosaic_project/scripts/
BASE_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__)) # Should be /home/ubuntu/mosaic_project
SCRIPTS_DIR = os.path.join(BASE_PROJECT_DIR, "scripts")

QUANTIZE_SCRIPT = os.path.join(SCRIPTS_DIR, "quantize_color_v3.py")
APPLY_CONTOUR_SCRIPT = os.path.join(SCRIPTS_DIR, "apply_good_contour_v4.py")
FILTER_THIN_SCRIPT = os.path.join(SCRIPTS_DIR, "filter_thin_lines_v5.py")
LAY_TESSERAE_SCRIPT = os.path.join(SCRIPTS_DIR, "lay_tesserae_contours_v6.py")
FILL_TESSERAE_SCRIPT = os.path.join(SCRIPTS_DIR, "fill_tesserae_areas_v7.py")

PYTHON_EXECUTABLE = "python3.11"

def run_script_step(script_path, args_list, step_name):
    """Runs a python script as a subprocess and logs its output."""
    command = [PYTHON_EXECUTABLE, script_path] + args_list
    logger.info(f"--- Running Step: {step_name} ---")
    logger.info(f"Executing: {' '.join(command)}")
    try:
        # Execute from the base project directory to ensure relative paths in scripts (if any) work as expected.
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=BASE_PROJECT_DIR)
        stdout, stderr = process.communicate(timeout=600) # 10 minutes timeout

        if stdout:
            logger.info(f"Output from {os.path.basename(script_path)}:\n{stdout.strip()}")
        if stderr:
            # Log critical errors as ERROR, others (like warnings or verbose info) as INFO
            stderr_strip = stderr.strip()
            if any(err_keyword in stderr_strip.lower() for err_keyword in ["error", "failed", "traceback", "exception"]):
                 logger.error(f"Error from {os.path.basename(script_path)}:\n{stderr_strip}")
            elif stderr_strip: # If there's any other stderr output
                 logger.info(f"Stderr/Info from {os.path.basename(script_path)}:\n{stderr_strip}")

        if process.returncode != 0:
            logger.error(f"{os.path.basename(script_path)} (Step: {step_name}) failed with return code {process.returncode}")
            return False
        logger.info(f"--- Step {step_name} completed successfully ---")
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout expired while running {os.path.basename(script_path)} for step {step_name}")
        if 'process' in locals() and process:
            process.kill()
        return False
    except Exception as e:
        logger.error(f"Exception running {os.path.basename(script_path)} for step {step_name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run the mosaic generation pipeline.")
    parser.add_argument("--input_image", required=True, help="Path to the input image.")
    parser.add_argument("--output_dir", required=True, help="Directory to save processed images and logs.")
    
    # Parameters for quantize_color_v3.py
    parser.add_argument("--num_colors", type=int, default=16, help="Number of colors for quantization (default: 16).")
    
    # Parameters for filter_thin_lines_v5.py
    parser.add_argument("--min_thickness", type=int, default=3, help="Minimum thickness for filtering thin lines in px (default: 3).")
    parser.add_argument("--min_area_after_erosion", type=int, default=5, help="Minimum area after erosion for filtering thin lines (default: 5).")

    # Parameters for lay_tesserae_contours_v6.py & fill_tesserae_areas_v7.py
    parser.add_argument("--tessera_width", type=int, default=10, help="Approximate width of a tessera in pixels (default: 10).")
    parser.add_argument("--tessera_height", type=int, default=15, help="Approximate height of a tessera in pixels (default: 15).")

    args = parser.parse_args()

    if not os.path.isfile(args.input_image):
        logger.error(f"Input image not found: {args.input_image}")
        return

    # Create a unique subdirectory for this run's outputs within the user-specified output_dir
    image_name_no_ext = os.path.splitext(os.path.basename(args.input_image))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_run_output_dir = os.path.join(args.output_dir, f"{image_name_no_ext}_{timestamp}")
    
    if not os.path.exists(current_run_output_dir):
        os.makedirs(current_run_output_dir)
        logger.info(f"Created output directory for this run: {current_run_output_dir}")
    
    # Setup file handler for logging for this specific run
    log_file_path = os.path.join(current_run_output_dir, f"pipeline_log_{timestamp}.log")
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler) # Add to root logger to capture all logs

    logger.info(f"Starting mosaic pipeline for image: {args.input_image}")
    logger.info(f"Output will be saved in: {current_run_output_dir}")

    # Define intermediate file paths
    quantized_image_path = os.path.join(current_run_output_dir, f"{image_name_no_ext}_1_quantized.png")
    good_contours_image_path = os.path.join(current_run_output_dir, f"{image_name_no_ext}_2_good_contours.png")
    canny_edge_map_path = os.path.join(current_run_output_dir, f"{image_name_no_ext}_2_canny_edge_map.png") # For apply_good_contour_v4.py artifact
    filtered_image_path = os.path.join(current_run_output_dir, f"{image_name_no_ext}_3_filtered.png")
    tesserae_contours_image_path = os.path.join(current_run_output_dir, f"{image_name_no_ext}_4_tess_contours.png")
    final_filled_tesserae_path = os.path.join(current_run_output_dir, f"{image_name_no_ext}_5_tess_filled.png")

    # --- Stage 1: Color Quantization ---
    quantize_args = [
        "--input_image", args.input_image,
        "--output_image", quantized_image_path,
        "--num_colors", str(args.num_colors)
    ]
    if not run_script_step(QUANTIZE_SCRIPT, quantize_args, "Color Quantization"):
        logger.error("Pipeline halting at Color Quantization stage.")
        return

    # --- Stage 2: Apply Good Contours ---
    apply_contour_args = [
        "--input_quantized_image", quantized_image_path,
        "--original_image", args.input_image, 
        "--output_contour_image", good_contours_image_path,
        "--output_canny_map", canny_edge_map_path 
    ]
    if not run_script_step(APPLY_CONTOUR_SCRIPT, apply_contour_args, "Apply Good Contours"):
        logger.error("Pipeline halting at Apply Good Contours stage.")
        return

    # --- Stage 3: Filter Thin Lines ---
    filter_thin_args = [
        "--input_image", good_contours_image_path,
        "--output_image", filtered_image_path,
        "--min_thickness", str(args.min_thickness),
        "--min_area_after_erosion", str(args.min_area_after_erosion)
    ]
    if not run_script_step(FILTER_THIN_SCRIPT, filter_thin_args, "Filter Thin Lines"):
        logger.error("Pipeline halting at Filter Thin Lines stage.")
        return

    # --- Stage 4: Lay Tesserae Contours ---
    lay_tesserae_args = [
        "--input_image", filtered_image_path, 
        "--output_image", tesserae_contours_image_path,
        "--tessera_width", str(args.tessera_width),
        "--tessera_height", str(args.tessera_height)
    ]
    if not run_script_step(LAY_TESSERAE_SCRIPT, lay_tesserae_args, "Lay Tesserae Contours"):
        logger.error("Pipeline halting at Lay Tesserae Contours stage.")
        return

    # --- Stage 5: Fill Tesserae Areas ---
    fill_tesserae_args = [
        "--input_image", filtered_image_path, 
        "--output_image", final_filled_tesserae_path,
        "--tessera_width", str(args.tessera_width),
        "--tessera_height", str(args.tessera_height)
    ]
    if not run_script_step(FILL_TESSERAE_SCRIPT, fill_tesserae_args, "Fill Tesserae Areas"):
        logger.error("Pipeline halting at Fill Tesserae Areas stage.")
        return

    logger.info(f"Mosaic pipeline completed successfully! Output in: {current_run_output_dir}")
    logger.info(f"Final image: {final_filled_tesserae_path}")
    logger.info(f"Log file: {log_file_path}")

if __name__ == "__main__":
    main()

