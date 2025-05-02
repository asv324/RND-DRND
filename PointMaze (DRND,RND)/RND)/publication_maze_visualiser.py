"""
Improved maze visualizer with black walls, grid overlay, and better legend placement.
Creates publication-quality visualizations of maze configurations.
"""
import numpy as np
import matplotlib.pyplot as plt
from maze_config import get_maze_config, list_available_mazes
import os

# Set better default styles for publication quality
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'

def visualize_maze(maze_type="umaze", save_path=None):
    """
    Visualize a maze configuration with improved styling for publications
    
    Args:
        maze_type (str): One of 'umaze', 'open', 'medium', 'large'
        save_path (str): Path to save the visualization (optional)
    """
    # Get maze configuration
    maze_config = get_maze_config(maze_type)
    
    # Extract maze map and positions
    maze_map = np.array(maze_config['maze_map'])
    start_pos = maze_config['fixed_start']
    goal_pos = maze_config['fixed_goal'] 
    
    # Scale factors for continuous coordinates to maze cells
    # In PointMaze, the maze is centered at the origin and each cell is 2x2 units
    maze_scaling = 2.0
    
    # Calculate grid position based on continuous coordinates
    # Add 4 to center around the origin, then divide by maze_scaling for cell position
    start_cell_x = int((start_pos[0] + 4) / maze_scaling)
    start_cell_y = int((start_pos[1] + 4) / maze_scaling)
    
    goal_cell_x = int((goal_pos[0] + 4) / maze_scaling)
    goal_cell_y = int((goal_pos[1] + 4) / maze_scaling)
    
    # Create figure and axis with higher resolution
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    
    # Plot maze map with black walls and white paths
    cmap = plt.cm.colors.ListedColormap(['white', 'black'])  # White paths, black walls
    ax.imshow(maze_map, cmap=cmap, interpolation='nearest')
    
    # Add grid overlay
    # First, get maze dimensions
    rows, cols = maze_map.shape
    
    # Add horizontal grid lines
    for i in range(rows + 1):
        ax.axhline(i - 0.5, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Add vertical grid lines
    for i in range(cols + 1):
        ax.axvline(i - 0.5, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Mark start position with nicer styling
    ax.plot(start_cell_y, start_cell_x, 
            marker='o', 
            markersize=20, 
            markerfacecolor='#4285F4',  # Blue
            markeredgecolor='white',
            markeredgewidth=1.5)
    
    # Mark goal position with nicer styling
    ax.plot(goal_cell_y, goal_cell_x, 
            marker='*', 
            markersize=25, 
            markerfacecolor='#EA4335',  # Red
            markeredgecolor='white',
            markeredgewidth=1.5)
    
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add legend in a better position with cleaner style
    # Create a custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4285F4', 
               markeredgecolor='white', markersize=10, label='Start'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#EA4335', 
               markeredgecolor='white', markersize=14, label='Goal')
    ]
    
    # Add the legend in bottom right (less cluttered)
    ax.legend(handles=legend_elements, loc='lower right', 
              frameon=True, framealpha=0.9, edgecolor='#CCCCCC',
              fontsize=11)
    
    # Add title
    ax.set_title(f"{maze_type.upper()} Maze", fontsize=16, fontweight='bold', pad=10)
    
    # Create info text box
    height, width = maze_map.shape
    info_text = (f"Environment ID: {maze_config['env_id']}\n"
                f"Maze Dimensions: {height}×{width}\n"
                f"Start Position: ({start_pos[0]:.1f}, {start_pos[1]:.1f})\n"
                f"Goal Position: ({goal_pos[0]:.1f}, {goal_pos[1]:.1f})")
    
    # Add info text in a box at the bottom
    plt.figtext(0.5, 0.01, info_text, 
                ha="center", va="bottom", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.5", 
                          facecolor="#F8F8F8", 
                          edgecolor="#CCCCCC", 
                          alpha=0.95))
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    
    # Save or display the figure
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"Saved visualization to {save_path}")
        return save_path
    else:
        plt.show()
        return None
    
def generate_all_visualizations(output_dir="maze_visualizations"):
    """Generate visualizations for all maze types"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate visualization for each maze type
    for maze_type in list_available_mazes():
        print(f"Generating visualization for {maze_type} maze...")
        save_path = os.path.join(output_dir, f"{maze_type}_maze.png")
        visualize_maze(maze_type, save_path)
    
    print(f"\nAll visualizations saved to '{output_dir}' directory")

if __name__ == "__main__":
    # Print available maze types
    print(f"Available maze types: {list_available_mazes()}")
    
    # Generate all visualizations
    generate_all_visualizations()
    
    # Optionally display one maze interactively
    if input("\nWould you like to display the UMaze visualization? (y/n): ").lower() == 'y':
        visualize_maze("umaze")