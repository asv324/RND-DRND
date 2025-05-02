import gymnasium as gym
import gymnasium_robotics

# Register gymnasium-robotics environments
gym.register_envs(gymnasium_robotics)

# Try to create a simple maze environment
env = gym.make("PointMaze_UMaze-v3")
print("Environment created successfully!")

# Close the environment
env.close()