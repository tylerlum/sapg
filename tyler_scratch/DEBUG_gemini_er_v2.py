import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# --- Corrected Data Definition (scaled to 640x360 image) ---

# 1. Bounding Box Data: [ymin, xmin, ymax, xmax] format
bounding_boxes = [
    {"box": [290, 120, 330, 190], "label": "Rubik's Cube"},
    {"box": [80, 520, 140, 595], "label": "Plates/Bowls"},
    {"box": [280, 170, 330, 470], "label": "Hammer"},
    {"box": [0, 0, 360, 400], "label": "Robotic arm"},
    {"box": [180, 530, 250, 620], "label": "Gripper/End effector"}
]

# 2. Sparse Keypoints Trajectory Data: [y, x] format
# Hammer Head Trajectory (Sparse)
head_trajectory_sparse = [
    [290, 470],  # Start Position (Resting near target)
    [100, 300],  # Pre-strike Position (Apex of swing)
    [290, 150],  # Impact Position (Striking Rubik's Cube)
    [320, 100]   # Follow-through Position (Post-impact)
]

# Hammer Handle Trajectory (Sparse)
handle_trajectory_sparse = [
    [320, 200],  # Start Position (Resting near target)
    [130, 180],  # Pre-strike Position (Apex of swing)
    [320, 30],   # Impact Position (Striking Rubik's Cube)
    [350, -20]   # Follow-through Position (Post-impact)
]

# 3. Dense Keypoints Trajectory Data: [y, x] format
# Hammer Head Trajectory (Dense) - interpolated points for a smooth arc
head_trajectory_dense = [
    [290, 470], [245, 420], [200, 370], [150, 335], [100, 300],
    [140, 260], [180, 220], [230, 185], [290, 150], [305, 125], [320, 100]
]

# Hammer Handle Trajectory (Dense) - interpolated points for a smooth arc
handle_trajectory_dense = [
    [320, 200], [285, 190], [250, 180], [190, 180], [130, 180],
    [170, 140], [210, 100], [260, 65], [320, 30], [335, 5], [350, -20]
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

        width = xmax - xmin
        height = ymax - ymin
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='g', facecolor='none')

        ax.add_patch(rect)
        ax.text(xmin, ymin - 10, label, color='g', fontsize=12, backgroundcolor='k')

    ax.set_title("1. Bounding Boxes on Objects (Corrected for 640x360)")
    ax.axis('off')
    plt.show()

def visualize_sparse_trajectory(img, head_traj, handle_traj):
    """Draws sparse keypoints and connections on the image."""
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(img)

    head_x = [p[1] for p in head_traj]
    head_y = [p[0] for p in head_traj]
    handle_x = [p[1] for p in handle_traj]
    handle_y = [p[0] for p in handle_traj]

    ax.plot(head_x, head_y, color='blue', linestyle='--', marker='o', markersize=8, label='Hammer Head Path')
    ax.plot(handle_x, handle_y, color='red', linestyle='--', marker='o', markersize=8, label='Hammer Handle Path')

    ax.text(head_x[0], head_y[0] - 10, "Start", color='blue', fontsize=12, backgroundcolor='w')
    ax.text(head_x[2] - 50, head_y[2] + 20, "Impact", color='blue', fontsize=12, backgroundcolor='w')
    ax.text(head_x[-1], head_y[-1] + 20, "End/Follow-through", color='blue', fontsize=12, backgroundcolor='w')

    ax.set_title("2. Sparse Keypoints Trajectory (Corrected for 640x360)")
    ax.axis('off')
    plt.show()

def visualize_dense_trajectory(img, head_traj, handle_traj):
    """Draws dense trajectory path on the image."""
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(img)

    head_x = [p[1] for p in head_traj]
    head_y = [p[0] for p in head_traj]
    handle_x = [p[1] for p in handle_traj]
    handle_y = [p[0] for p in handle_traj]

    ax.plot(head_x, head_y, color='blue', linewidth=3, label='Hammer Head Path')
    ax.plot(handle_x, handle_y, color='red', linewidth=3, label='Hammer Handle Path')

    ax.set_title("3. Dense Trajectory Path (Corrected for 640x360)")
    ax.axis('off')
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    img = load_image()
    if img:
        visualize_bounding_boxes(img, bounding_boxes)
        visualize_sparse_trajectory(img, head_trajectory_sparse, handle_trajectory_sparse)
        visualize_dense_trajectory(img, head_trajectory_dense, handle_trajectory_dense)