import cv2
import numpy as np
import time
import psutil
import os
from concurrent.futures import ProcessPoolExecutor

def apply_sobel(region):
    """Applies the Sobel filter to a 3x3 region and returns the gradient magnitude."""
    # Sobel kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    # Calculate gradients
    gx = np.sum(region * sobel_x)
    gy = np.sum(region * sobel_y)
    gradient_magnitude = np.sqrt(gx**2 + gy**2)
    
    return gradient_magnitude

def process_chunk(image_chunk):
    """Processes a chunk of the image with the Sobel operator."""
    rows, cols = image_chunk.shape
    result_chunk = np.zeros((rows, cols), dtype=np.float32)
    
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            region = image_chunk[i - 1:i + 2, j - 1:j + 2]
            result_chunk[i, j] = apply_sobel(region)
    
    return result_chunk

def sobel_edge_detection_parallel(image, num_workers=4):
    """
    Perform Sobel edge detection on the image using parallel processing on CPU.
    Saves the result as an image file in the current directory.
    """
    # Split the image into chunks along the rows for parallel processing
    chunk_height = image.shape[0] // num_workers
    image_chunks = [image[i * chunk_height: (i + 1) * chunk_height + 2] for i in range(num_workers)]
    
    # Process each chunk in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        processed_chunks = list(executor.map(process_chunk, image_chunks))  # Convert generator to list
    
    # Concatenate processed chunks to form the complete result
    gradient_magnitude = np.vstack(processed_chunks)
    
    # Normalize and convert to uint8 for saving
    gradient_magnitude = np.clip(gradient_magnitude / gradient_magnitude.max() * 255, 0, 255).astype(np.uint8)
    cv2.imwrite('edge_detected_image_parallel_cpu.jpg', gradient_magnitude)

    return gradient_magnitude

if __name__ == '__main__':
    # Load the image in grayscale
    image = cv2.imread('pic.jpg', cv2.IMREAD_GRAYSCALE)

    # Measure the start time
    start_time = time.time()

    # Track CPU and memory usage before processing
    process = psutil.Process(os.getpid())
    initial_cpu_usage = process.cpu_percent(interval=None)
    initial_memory_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB

    # Perform parallel edge detection
    edge_detected_image = sobel_edge_detection_parallel(image)

    # Measure end time and calculate total time taken
    end_time = time.time()
    time_taken = end_time - start_time

    # Track CPU and memory usage after processing
    final_cpu_usage = process.cpu_percent(interval=None)
    final_memory_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB

    # Track average CPU usage during processing
    cpu_usage_during_execution = process.cpu_percent(interval=0.1)  # Initial usage snapshot with an interval

    # Perform parallel edge detection
    edge_detected_image = sobel_edge_detection_parallel(image)

    # Measure end time and calculate total time taken
    end_time = time.time()
    time_taken = end_time - start_time

    # Track final CPU usage
    final_cpu_usage = process.cpu_percent(interval=0.1)  # End usage snapshot with an interval

    # Prepare log content, now showing initial, during, and final CPU usage
    log_content = (
        "CPU PARALLEL\n"
        f"Time taken for edge detection: {time_taken:.2f} seconds\n"
        f"Initial CPU usage: {initial_cpu_usage:.2f}%\n"
        f"CPU usage during execution: {cpu_usage_during_execution:.2f}%\n"
        f"Final CPU usage: {final_cpu_usage:.2f}%\n"
        f"Initial memory usage: {initial_memory_usage:.2f} MB\n"
        f"Final memory usage: {final_memory_usage:.2f} MB\n"
        f"Memory used during execution: {final_memory_usage - initial_memory_usage:.2f} MB\n"
        "--------------------------------------------------\n"
    )


    # Write the log to file
    with open("cpu_execution_log.txt", "a") as log_file:
        log_file.write(log_content)

    print("Edge-detected image saved as 'edge_detected_image_parallel_cpu.jpg'")
    print("Execution statistics saved in 'cpu_parallel_execution_log.txt'")
