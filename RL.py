from typing import Optional
import numpy as np
import gymnasium as gym


class CursorImageEnv(gym.Env):
    def __init__(self, full_image_size=(256, 256, 180), window_size = (64, 64, 32)):

        self.full_image_size = full_image_size
        self.window_size = window_size
        self.cursor_position = np.array([full_image_size[0] - (window_size[0]/2), full_image_size[1] - (window_size[1]/2)], full_image_size[2] - (window_size[2]/2)) # Assume the window is slid across 2D? 

        # agent and target location still required?

        # Observations are 256 x 256 x 180 images (e.g., medical imaging or volumetric data)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=window_size, dtype=np.uint8
        )

        # Action space is discrete, with 6 possible actions: up, down, left, right, forward, backward
        self.action_space = gym.spaces.Discrete(6)

        # load the classifier model

    def _get_obs(self):
        """Crop the image around the cursor."""
        # Assumes that cursor is in the center of the crop window
        half_crop = np.array(self.crop_size[:2]) // 2
        start = np.maximum(self.cursor_position - half_crop, 0) # Ensure we don't go out of bounds
        end = np.minimum(start + self.crop_size[:2], self.full_image_size[:2]) # ignores depth dimension

        # Adjust start if we're near the edge of the image
        start = np.maximum(end - self.crop_size[:2], 0)

        # Crop in the first two dimensions; keep depth (z-axis) fully intact
        crop = self.image[
            start[0]:end[0],
            start[1]:end[1],
            :
        ]
        return crop
    
    def reset(self):
        self.cursor_position = np.array([64, 64, 32]) # random range 
        return self._get_obs()
    
    def step(self, action):
        if action == 0: 
            self.cursor_position = cursor_position + np.array([0, 0, 4])

        # if statements for each direction (up, down, left, right, forward, backward) left right, forward, backward
        # get new cursor position after each direction
        # user cursor position to crop the image == observation
        # pass through crop through the classifier --> reward
        self.cursor_position = np.clip(self.cursor_position + direction, 0, self.full_image_size[:2])
        obs = self._get_obs()
        return obs, 0.0, False, {}