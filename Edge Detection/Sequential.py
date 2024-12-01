"""import cv2
import numpy as np
import time
import psutil
import os

def sobel_edge_detection(image):
    """
"""
    Apply Sobel edge detection to an input grayscale image.
    Saves the result as an image file in the current directory.
    
    Args:
    - image: np.array, grayscale image to process.
    
    Returns:
    - gradient_magnitude: np.array, edge-detected image.
"""    """
    # Sobel kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Get image dimensions
    rows, cols = image.shape

    # Initialize gradient images
    gradient_x = np.zeros((rows, cols), dtype=np.float32)
    gradient_y = np.zeros((rows, cols), dtype=np.float32)
    gradient_magnitude = np.zeros((rows, cols), dtype=np.float32)

    # Sequentially apply the Sobel operator
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # Extract the 3x3 region around each pixel
            region = image[i - 1:i + 2, j - 1:j + 2]

            # Apply the Sobel operator in both x and y directions
            gx = np.sum(region * sobel_x)
            gy = np.sum(region * sobel_y)

            # Store the gradients
            gradient_x[i, j] = gx
            gradient_y[i, j] = gy

            # Compute the gradient magnitude (edge strength)
            gradient_magnitude[i, j] = np.sqrt(gx**2 + gy**2)

    # Normalize and convert to uint8 for saving
    gradient_magnitude = np.clip(gradient_magnitude / gradient_magnitude.max() * 255, 0, 255).astype(np.uint8)
    cv2.imwrite('edge_detected_image.jpg', gradient_magnitude)

    return gradient_magnitude

# Load the image in grayscale
image = cv2.imread('pic.jpg', cv2.IMREAD_GRAYSCALE)

# Measure the start time
start_time = time.time()

# Track CPU and memory usage before processing
process = psutil.Process(os.getpid())
initial_cpu_usage = process.cpu_percent(interval=None)
initial_memory_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB

# Perform edge detection
edge_detected_image = sobel_edge_detection(image)

# Measure end time and calculate total time taken
end_time = time.time()
time_taken = end_time - start_time

# Track CPU and memory usage after processing
final_cpu_usage = process.cpu_percent(interval=None)
final_memory_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB

# Prepare log content
log_content = (
    "SEQUENTIAL\n"
    f"Time taken for edge detection: {time_taken:.2f} seconds\n"
    f"Initial CPU usage: {initial_cpu_usage:.2f}%\n"
    f"Final CPU usage: {final_cpu_usage:.2f}%\n"
    f"Initial memory usage: {initial_memory_usage:.2f} MB\n"
    f"Final memory usage: {final_memory_usage:.2f} MB\n"
    f"Memory used during execution: {final_memory_usage - initial_memory_usage:.2f} MB\n"
    "--------------------------------------------------\n"
)

# Write the log to file
with open("sequential_execution_log.txt", "a") as log_file:
    log_file.write(log_content)

print("Edge-detected image saved as 'edge_detected_image.jpg'")
print("Execution statistics saved in 'sequential_execution_log.txt'")
"""


import cv2
import numpy as np
import time
import psutil
import os

def sobel_edge_detection(image):
    """
    Apply Sobel edge detection to an input grayscale image.
    
    Args:
    - image: np.array, grayscale image to process.
    
    Returns:
    - gradient_magnitude: np.array, edge-detected image.
    """
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    rows, cols = image.shape
    gradient_x = np.zeros((rows, cols), dtype=np.float32)
    gradient_y = np.zeros((rows, cols), dtype=np.float32)
    gradient_magnitude = np.zeros((rows, cols), dtype=np.float32)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            region = image[i - 1:i + 2, j - 1:j + 2]
            gx = np.sum(region * sobel_x)
            gy = np.sum(region * sobel_y)
            gradient_x[i, j] = gx
            gradient_y[i, j] = gy
            gradient_magnitude[i, j] = np.sqrt(gx**2 + gy**2)

    gradient_magnitude = np.clip(gradient_magnitude / gradient_magnitude.max() * 255, 0, 255).astype(np.uint8)
    return gradient_magnitude

# List of 50 image filenames
image_filenames = [f'image_{i}.jpg' for i in range(1, 51)]

# Process each image
for idx, filename in enumerate(image_filenames):
    # Load the image in grayscale
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Image '{filename}' could not be loaded. Skipping.")
        continue
    
    start_time = time.time()
    process = psutil.Process(os.getpid())
    initial_cpu_usage = process.cpu_percent(interval=None)
    initial_memory_usage = process.memory_info().rss / (1024 * 1024)  # MB

    # Perform edge detection
    edge_detected_image = sobel_edge_detection(image)

    # Measure end time and calculate total time taken
    end_time = time.time()
    time_taken = end_time - start_time

    final_cpu_usage = process.cpu_percent(interval=None)
    final_memory_usage = process.memory_info().rss / (1024 * 1024)  # MB

    # Save the edge-detected image
    output_filename = f'edge_detected_{filename}'
    cv2.imwrite(output_filename, edge_detected_image)

    # Prepare log content
    log_content = (
        f"Image {idx + 1}: {filename}\n"
        f"Time taken for edge detection: {time_taken:.2f} seconds\n"
        f"Initial CPU usage: {initial_cpu_usage:.2f}%\n"
        f"Final CPU usage: {final_cpu_usage:.2f}%\n"
        f"Initial memory usage: {initial_memory_usage:.2f} MB\n"
        f"Final memory usage: {final_memory_usage:.2f} MB\n"
        f"Memory used during execution: {final_memory_usage - initial_memory_usage:.2f} MB\n"
        "--------------------------------------------------\n"
    )

    # Write the log to file
    with open("sequential_execution_log.txt", "a") as log_file:
        log_file.write(log_content)

    print(f"Processed '{filename}' - edge-detected image saved as '{output_filename}'")
    print(f"Execution statistics for '{filename}' saved in 'sequential_execution_log.txt'")
