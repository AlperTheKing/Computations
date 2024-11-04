import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import sys

def load_image(file_path):
    """
    Loads an image from the specified file path.

    Args:
        file_path (str): Path to the image file.

    Returns:
        image (numpy.ndarray): Loaded image in BGR format.

    Raises:
        FileNotFoundError: If the image cannot be loaded.
    """
    image = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Error: Image '{file_path}' not found. Please check the file path.")
    return image

def crop_sub_images(image, positions):
    """
    Crops sub-images from the main image based on provided coordinates.

    Args:
        image (numpy.ndarray): The main image.
        positions (dict): Dictionary mapping labels to coordinates (x1, y1, x2, y2).

    Returns:
        sub_images (dict): Dictionary mapping labels to cropped sub-images.
    """
    sub_images = {}
    for label, (x1, y1, x2, y2) in positions.items():
        # Validate coordinates
        if x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
            print(f"Warning: Coordinates for sub-image '{label}' are out of bounds. Skipping.")
            continue
        sub_img = image[y1:y2, x1:x2].copy()  # Use .copy() to ensure a separate memory block
        sub_images[label] = sub_img
    return sub_images

def rotate_image(image, angle):
    """
    Rotates an image by the specified angle.

    Args:
        image (numpy.ndarray): Image to rotate.
        angle (int): Angle in degrees. Must be one of [0, 90, 180, 270].

    Returns:
        rotated (numpy.ndarray): Rotated image.
    """
    if angle == 0:
        return image
    elif angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        raise ValueError("Angle must be one of [0, 90, 180, 270].")

def compute_histogram(image, bins=(16, 16, 16)):
    """
    Computes a normalized HSV color histogram for an image.

    Args:
        image (numpy.ndarray): Image in BGR format.
        bins (tuple): Number of bins for each HSV channel.

    Returns:
        hist (numpy.ndarray): Flattened and normalized histogram.
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv_image],
        [0, 1, 2],
        None,
        bins,
        [0, 180, 0, 256, 0, 256]
    )
    cv2.normalize(hist, hist)
    return hist.flatten()

def select_best_rotation(rotations, histograms):
    """
    Selects the best rotation based on histogram entropy.

    Args:
        rotations (list): List of rotated images.
        histograms (list): Corresponding list of histograms.

    Returns:
        best_index (int): Index of the best rotation.
    """
    entropies = []
    for hist in histograms:
        # Calculate entropy
        entropy = -np.sum(hist * np.log(hist + 1e-10))  # Avoid log(0)
        entropies.append(entropy)
    best_index = np.argmin(entropies)  # Lower entropy implies more uniform histogram
    return best_index

def sharpen_image(image):
    """
    Applies a sharpening filter to the image.

    Args:
        image (numpy.ndarray): Input image in BGR format.

    Returns:
        sharpened (numpy.ndarray): Sharpened image.
    """
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def process_sub_images(sub_images, resize_dim=(300, 300), apply_sharpen=False):
    """
    Processes sub-images: resizing, rotating, computing histograms, and optionally sharpening.

    Args:
        sub_images (dict): Dictionary mapping labels to cropped sub-images.
        resize_dim (tuple): Dimensions to resize sub-images to (width, height).
        apply_sharpen (bool): Whether to apply sharpening to sub-images.

    Returns:
        processed_images (dict): Dictionary mapping labels to the best-rotated images.
        histograms (dict): Dictionary mapping labels to their histograms.
    """
    processed_images = {}
    histograms = {}
    for label, img in sub_images.items():
        # Resize image with high-quality interpolation
        resized_img = cv2.resize(img, resize_dim, interpolation=cv2.INTER_LANCZOS4)

        # Optionally apply sharpening
        if apply_sharpen:
            resized_img = sharpen_image(resized_img)

        # Generate rotations
        angles = [0, 90, 180, 270]
        rotations = [rotate_image(resized_img, angle) for angle in angles]

        # Compute histograms for each rotation
        hist_list = [compute_histogram(rot) for rot in rotations]

        # Select the best rotation based on entropy
        best_idx = select_best_rotation(rotations, hist_list)
        best_rotation = rotations[best_idx]
        best_hist = hist_list[best_idx]

        # Store results
        processed_images[label] = best_rotation
        histograms[label] = best_hist
    return processed_images, histograms

def compare_histograms(histograms, labels):
    """
    Compares histograms and computes pairwise Chi-Square distances.

    Args:
        histograms (dict): Dictionary mapping labels to histograms.
        labels (list): List of labels.

    Returns:
        differences (dict): Dictionary mapping label pairs to Chi-Square distance scores.
    """
    differences = {}
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            label1, label2 = labels[i], labels[j]
            hist1, hist2 = histograms[label1], histograms[label2]
            score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
            differences[f"{label1}-{label2}"] = score
    return differences

def identify_most_different(differences, labels):
    """
    Identifies the most different sub-image based on average Chi-Square distances.

    Args:
        differences (dict): Dictionary mapping label pairs to Chi-Square distance scores.
        labels (list): List of labels.

    Returns:
        most_different (str): Label of the most different sub-image.
    """
    average_scores = {label: 0 for label in labels}
    count = {label: 0 for label in labels}

    for pair, score in differences.items():
        label1, label2 = pair.split('-')
        average_scores[label1] += score
        average_scores[label2] += score
        count[label1] += 1
        count[label2] += 1

    # Compute average scores
    for label in average_scores:
        if count[label] > 0:
            average_scores[label] /= count[label]
        else:
            average_scores[label] = float('inf')  # In case of no comparisons

    # Identify the label with the highest average distance (most different)
    most_different = max(average_scores, key=average_scores.get)
    return most_different

def display_sub_images(processed_images, labels, most_different):
    """
    Displays sub-images in a grid, highlighting the most different one.

    Args:
        processed_images (dict): Dictionary mapping labels to images.
        labels (list): List of labels.
        most_different (str): Label of the most different sub-image.
    """
    grid_rows = 2
    grid_cols = 3
    fig, axs = plt.subplots(grid_rows, grid_cols, figsize=(18, 12))
    axs = axs.flatten()

    for i, label in enumerate(labels):
        if label not in processed_images:
            axs[i].axis('off')
            axs[i].set_title(f"{label} (Skipped)", fontsize=14)
            continue
        img = processed_images[label]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if label == most_different:
            # Add a red border
            border_size = 20
            img_rgb = cv2.copyMakeBorder(
                img_rgb,
                top=border_size,
                bottom=border_size,
                left=border_size,
                right=border_size,
                borderType=cv2.BORDER_CONSTANT,
                value=[255, 0, 0]  # Red color
            )
        axs[i].imshow(img_rgb)
        axs[i].set_title(f"Label: {label}", fontsize=16)
        axs[i].axis('off')

    plt.suptitle(f"The most different sub-image is: {most_different}", fontsize=20, color='red')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def main():
    # -------------------- Configuration -------------------- #
    image_path = 'EFF141A4-DDC8-423A-988C-D4E4C9FB3931.jpeg'  # Your image file
    resize_dimensions = (300, 300)  # Increased dimensions for better quality
    apply_sharpening = True  # Set to True to apply image sharpening
    sub_image_positions = {
        "A": (45, 537, 430, 920),
        "B": (453, 537, 837, 905),
        "C": (861, 522, 1262, 906),
        "D": (56, 953, 437, 1329),
        "E": (470, 947, 845, 1322),
        "F": (879, 936, 1275, 1323)
    }
    labels = ["A", "B", "C", "D", "E", "F"]
    histogram_bins = (16, 16, 16)  # Increased bins for higher histogram resolution
    # -------------------------------------------------------- #

    try:
        # Load the main image
        image = load_image(image_path)
        print(f"Loaded image '{image_path}' with shape {image.shape}.")

        # Crop sub-images
        sub_images = crop_sub_images(image, sub_image_positions)
        if not sub_images:
            print("Error: No valid sub-images were cropped. Exiting.")
            sys.exit(1)
        print(f"Successfully cropped {len(sub_images)} sub-images.")

        # Process sub-images: resize, rotate, compute histograms, and optionally sharpen
        processed_images, histograms = process_sub_images(
            sub_images,
            resize_dim=resize_dimensions,
            apply_sharpen=apply_sharpening
        )
        print("Processed sub-images with enhanced quality.")

        # Ensure enough sub-images for comparison
        processed_labels = list(processed_images.keys())
        if len(processed_labels) < 2:
            print("Error: Need at least two sub-images to compare. Exiting.")
            sys.exit(1)

        # Compare histograms
        differences = compare_histograms(histograms, processed_labels)
        print("Computed pairwise histogram differences.")

        # Identify the most different sub-image
        most_different = identify_most_different(differences, processed_labels)
        print(f"The most different sub-image is: {most_different}")

        # Display sub-images
        display_sub_images(processed_images, labels, most_different)

    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    except ValueError as ve:
        print(f"Value Error: {ve}")
        sys.exit(1)
    except Exception as ex:
        print(f"An unexpected error occurred: {ex}")
        sys.exit(1)

if __name__ == "__main__":
    main()