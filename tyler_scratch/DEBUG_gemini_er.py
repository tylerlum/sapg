import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# --- Data Definition (based on previous responses) ---

# 1. Bounding Box Data: [ymin, xmin, ymax, xmax] format
bounding_boxes = [
    {"box": [753, 413, 903, 485], "label": "Rubik's Cube"},
    {"box": [435, 788, 564, 939], "label": "Plates/Bowls"},
    {"box": [648, 452, 775, 663], "label": "Hammer"},
    {"box": [0, 0, 751, 627], "label": "Robotic arm"},
    {"box": [182, 530, 258, 627], "label": "Gripper/End effector"}
]

# 2. Sparse Keypoints Trajectory Data: [y, x] format
# Hammer Head Trajectory (Sparse)
head_trajectory_sparse = [
    [703, 650],  # Start Position (Resting on Cube)
    [500, 550],  # Pre-strike Position (Lifted)
    [650, 480],  # Mid-swing Position (Downward Arc)
    [750, 420],  # Impact Position (Striking)
    [800, 400]   # Follow-through Position (Post-impact)
]

# Hammer Handle Trajectory (Sparse)
handle_trajectory_sparse = [
    [735, 506],  # Start Position (Resting on Cube)
    [532, 406],  # Pre-strike Position (Lifted)
    [682, 336],  # Mid-swing Position (Downward Arc)
    [782, 276],  # Impact Position (Striking)
    [832, 256]   # Follow-through Position (Post-impact)
]

# 3. Dense Keypoints Trajectory Data: [y, x] format
# Hammer Head Trajectory (Dense)
head_trajectory_dense = [
    [703, 650], [652, 625], [602, 600], [551, 575], [500, 550],
    [538, 533], [575, 515], [613, 498], [650, 480], [675, 465],
    [700, 450], [725, 435], [750, 420], [767, 413], [783, 407],
    [800, 400]
]

# Hammer Handle Trajectory (Dense)
handle_trajectory_dense = [
    [735, 506], [684, 481], [634, 456], [583, 431], [532, 406],
    [570, 389], [607, 371], [645, 354], [682, 336], [707, 321],
    [732, 306], [757, 291], [782, 276], [799, 269], [815, 263],
    [832, 256]
]

# --- Visualization Functions ---

def load_image(image_path="hammer.png"):
    """Loads the original image."""
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: Image file not found at '{image_path}'. Please replace '{image_path}' with the correct path to your image.")
        return None
    return img

def visualize_bounding_boxes(img, boxes):
    """Draws bounding boxes on the image."""
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(img)

    for box_data in boxes:
        ymin, xmin, ymax, xmax = box_data['box']
        label = box_data['label']

        # Create a rectangle patch [ymin, xmin, ymax, xmax] -> (xmin, ymin, width, height)
        width = xmax - xmin
        height = ymax - ymin
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='g', facecolor='none')

        # Add the patch to the axes
        ax.add_patch(rect)

        # Add text label
        ax.text(xmin, ymin - 10, label, color='g', fontsize=12, backgroundcolor='k')

    ax.set_title("1. Bounding Boxes on Objects")
    ax.axis('off')

def visualize_sparse_trajectory(img, head_traj, handle_traj):
    """Draws sparse keypoints and connections on the image."""
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(img)

    # Convert [y, x] format to (x, y) for Matplotlib plotting
    head_x = [p[1] for p in head_traj]
    head_y = [p[0] for p in head_traj]
    handle_x = [p[1] for p in handle_traj]
    handle_y = [p[0] for p in handle_traj]

    # Plot hammer head path (blue line, blue dots)
    ax.plot(head_x, head_y, color='blue', linestyle='--', marker='o', markersize=8, label='Hammer Head Path')
    # Plot hammer handle path (red line, red dots)
    ax.plot(handle_x, handle_y, color='red', linestyle='--', marker='o', markersize=8, label='Hammer Handle Path')

    # Add labels for key points
    ax.text(head_x[0], head_y[0] - 10, "Start", color='blue', fontsize=12, backgroundcolor='w')
    ax.text(head_x[4], head_y[4] + 20, "End/Follow-through", color='blue', fontsize=12, backgroundcolor='w')
    ax.text(head_x[3] - 50, head_y[3] + 20, "Impact", color='blue', fontsize=12, backgroundcolor='w')

    ax.set_title("2. Sparse Keypoints Trajectory")
    ax.axis('off')

def visualize_dense_trajectory(img, head_traj, handle_traj):
    """Draws dense trajectory path on the image."""
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(img)

    # Convert [y, x] format to (x, y) for Matplotlib plotting
    head_x = [p[1] for p in head_traj]
    head_y = [p[0] for p in head_traj]
    handle_x = [p[1] for p in handle_traj]
    handle_y = [p[0] for p in handle_traj]

    # Plot dense path for hammer head (blue line)
    ax.plot(head_x, head_y, color='blue', linewidth=3, label='Hammer Head Path')
    # Plot dense path for hammer handle (red line)
    ax.plot(handle_x, handle_y, color='red', linewidth=3, label='Hammer Handle Path')

    ax.set_title("3. Dense Trajectory Path")
    ax.axis('off')

# --- Main Execution ---
if __name__ == "__main__":
    # Load image (ensure 'original_image.png' is in the same directory or provide full path)
    img = load_image()
    if img:
        # 1. Visualize Bounding Boxes
        visualize_bounding_boxes(img, bounding_boxes)

        # 2. Visualize Sparse Trajectory Keypoints
        visualize_sparse_trajectory(img, head_trajectory_sparse, handle_trajectory_sparse)

        # 3. Visualize Dense Trajectory Path
        visualize_dense_trajectory(img, head_trajectory_dense, handle_trajectory_dense)

        plt.show()