from typing import Optional
import numpy as np
import gymnasium as gym
import torch
from new_file_copy import Net
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
import nibabel as nib

# action space for RL agents different? 
class CursorImageEnv(gym.Env):
    def __init__(self, full_image_size=(256, 256, 180), window_size = (64, 64, 32)):
        # Define image and window size
        self.full_image_size = full_image_size
        self.window_size = window_size
        self.image = nib.load('liver_125.nii.gz').get_fdata()

        # Observations are 256 x 256 x 180 images (e.g., medical imaging or volumetric data)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=window_size, dtype=np.uint8
        )

        # Action space is discrete, with 6 possible actions: up, down, left, right, forward, backward
        self.action_space = gym.spaces.Discrete(6)

        # Load the classifier model
        self.classifier = Net()
        self.classifier.load_state_dict(torch.load("model_weights.pth"))
        self.classifier.eval()  # Set the model to evaluation mode

        self.competitor = None
        
    def _get_obs(self, cursor_position): # modify to include depth
        """Crop the image around the cursor."""
        # Assumes that cursor is in the center of the crop window
        half_crop = np.array(self.window_size) // 2
        start = np.maximum(cursor_position - half_crop, 0) # Ensure we don't go out of bounds
        end = np.minimum(start + self.crop_size, self.full_image_size) # Ensure we don't go out of bounds

        # Crop in the first two dimensions; keep depth (z-axis) fully intact
        crop = self.image[
            start[0]:end[0],
            start[1]:end[1],
            start[2]:end[2]
        ]
        return crop
    
    def reset(self):
        self.cursor_position_agent1 = np.array([self.image_size[0]//2, self.image_size[1]//2, self.image_size[2]//2])
        self.cursor_position_agent2 = np.array([self.image_size[0]//2, self.image_size[1]//2, self.image_size[2]//2])
        return self._get_obs(self.cursor_position_agent1)
    
    def _get_reward(self, obs):
        with torch.no_grad():
            outputs = self.classifier(obs)
            prediction = torch.softmax(outputs, dim=1)
            reward = prediction[0, 1].item()
        return reward
    
    def update_cursor_position(self, cursor_position, action):
        if action == 0: 
            self.cursor_position = cursor_position + np.array([0, 0, 4])
        if action == 1:
            self.cursor_position = cursor_position + np.array([0, 0, -4])
        if action == 2:
            self.cursor_position = cursor_position + np.array([0, 4, 0])
        if action == 3:
            self.cursor_position = cursor_position + np.array([0, -4, 0])
        if action == 4:
            self.cursor_position = cursor_position + np.array([4, 0, 0])
        if action == 5:
            self.cursor_position = cursor_position + np.array([-4, 0, 0])
        return self.cursor_position
    
    def step(self, action):
        # Update cursor position
        self.cursor_position_agent1 = self.update_cursor_position(self.cursor_position_agent1, action)

        # Get observations
        obs_agent1 = self._get_obs(self.cursor_position_agent1)

        obs_agent2 = self._get_obs(self.cursor_position_agent2)
        action_agent2 = self.competitor.predict(obs_agent2)[0]

        self.cursor_position_agent2 = self.update_cursor_position(self.cursor_position_agent2, action_agent2)
        new_obs_agent2 = self._get_obs(self.cursor_position_agent2)

        # Get rewards
        reward_agent1 = self._get_reward(obs_agent1)
        reward_agent2 = self._get_reward(new_obs_agent2)

        if reward_agent1 > reward_agent2:
            reward_agent1 = 1
            reward_agent2 = -1 
        
        if reward_agent1 < reward_agent2:
            reward_agent1 = -1
            reward_agent2 = 1
            
        if reward_agent1 or reward_agent2 > 0.9:
            done = True
        else:
            done = False

        return obs_agent1, reward_agent1, done, {}

    def set_competitor(self, new_competitor):
        self.competitor = new_competitor

class dummy_func():
    def __init__(self, env):
        self.env = env

    def predict(self, inputs):
        return [np.zeros(self.env.action_space.shape), 6]
        
def env_creator():
    env = CursorImageEnv()
    env.competitor = dummy_func
    return env

vec_env = DummyVecEnv([env_creator])

model = PPO("MlpPolicy", vec_env, verbose=2)

competitor_update_frequeny = 2 # every 2 steps
num_of_interations = 32

for iteration in range(num_of_interations):
    model.learn(competitor_update_frequeny)
    vec_env.env_method("set_competitor", model)





