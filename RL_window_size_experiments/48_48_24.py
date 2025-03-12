import numpy as np
import os
import gym
import torch
from net_global import Net
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
import nibabel as nib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
os.environ['CUDA_VISIBLE_DEVICES']='-1'


class CursorImageEnv(gym.Env):
    def __init__(self, full_image_size=(256, 256, 180), window_size=(48, 48, 24)):
        # Define image and window size
        self.window_size = window_size
        self.image = nib.load('/raid/candi/catalina/Task03_Liver/imagesTr/liver_5.nii.gz').get_fdata()
        self.full_image_size = full_image_size
        label = nib.load('/raid/candi/catalina/Task03_Liver/labelsTr/liver_5.nii.gz').get_fdata()
        label[label == 1] = 0
        label[label == 2] = 1
        self.label = label

        # Observations are 256 x 256 x 180 images (e.g., medical imaging or volumetric data)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=self.full_image_size, dtype=np.uint8
        )

        # Action space is discrete, with 6 possible actions: up, down, left, right, forward, backward
        self.action_space = gym.spaces.Discrete(6)

        # Load the classifier model
        self.classifier = Net()
        self.classifier.load_state_dict(torch.load("ex4_24in_weights.pth"))
        self.classifier.eval()  # Set the model to evaluation mode

        self.prediction = None
        self.competitor = None
    
        self.accumulated_predictions = np.zeros(self.full_image_size, dtype=np.float32)
        
    def resize_image(self, img_data):
        # Define target shape for the resized image
        target_shape = (256, 256, 180)
        
        # Convert input to a PyTorch tensor and add batch and channel dimensions
        img_tensor = torch.tensor(img_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Resize using torch.nn.functional.interpolate (trilinear interpolation)
        resized_tensor = F.interpolate(img_tensor, size=target_shape, mode='nearest')

        resized_data = resized_tensor.squeeze()  # Tensor Shape: (256, 256, 180)
        return np.array(resized_data)

    def _get_obs(self, cursor_position): # modify to include depth
        """Crop the image around the cursor."""
        # Assumes that cursor is in the center of the crop window
        start = np.clip(cursor_position, np.array([1, 1, 1]), np.array(self.full_image_size) - np.array(self.window_size) - np.array([1, 1, 1])) # Ensure we don't go out of bounds
        end = start + np.array(self.window_size) # Ensure we don't go out of bounds

        # Crop in the first two dimensions; keep depth (z-axis) fully intact
        crop = self.image[
            start[0]:end[0],
            start[1]:end[1],
            start[2]:end[2]
        ]

        resized_data = self.resize_image(crop)  # Tensor Shape: (256, 256, 180)
        return resized_data
    
    def reset(self):
        self.cursor_position_agent1 = np.array([self.full_image_size[0]//2, self.full_image_size[1]//2, self.full_image_size[2]//2])
        self.cursor_position_agent2 = np.array([self.full_image_size[0]//2, self.full_image_size[1]//2, self.full_image_size[2]//2])
        self.accumulated_predictions = np.zeros(self.full_image_size, dtype=np.float32)
        return np.array(self._get_obs(self.cursor_position_agent1))
    
    def dice_score(self, pred, target):
        pred = np.array(pred).astype(bool)
        target = target.astype(bool)
        resized_target = self.resize_image(target)

        intersection = np.logical_and(pred, resized_target).sum()
        dice = 2. * intersection / (pred.sum() + resized_target.sum())
        return dice

    def _get_reward(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # Tensor Shape: (1, 1, 256, 256, 180)
        with torch.no_grad():
            outputs = self.classifier(obs_tensor)
            self.prediction = torch.softmax(outputs, dim=1)
            reward = self.prediction[0, 1].item()
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

        print('step')

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

        done = reward_agent1 > 0.3 or reward_agent2 > 0.3
        
        if reward_agent1 >= reward_agent2:
            new_reward_agent1 = 1

        elif reward_agent1 < reward_agent2:
            new_reward_agent1 = -1

        self.reward_to_output = new_reward_agent1

        # Accumulate predictions
        self.accumulated_predictions[self.cursor_position_agent1[0]:self.cursor_position_agent1[0]+self.window_size[0],
                                     self.cursor_position_agent1[1]:self.cursor_position_agent1[1]+self.window_size[1],
                                     self.cursor_position_agent1[2]:self.cursor_position_agent1[2]+self.window_size[2]] += self.prediction[0, 1].cpu().numpy()
        
        return np.array(obs_agent1), new_reward_agent1, done, {}

    def compute_final_dice_score(self):
        thresholded_predictions = self.accumulated_predictions > 0.5
        return self.dice_score(thresholded_predictions, self.label)

    def set_competitor(self, new_competitor):
        self.competitor = new_competitor

test_env = CursorImageEnv()

class dummy_func():
    def __init__():
        pass

    def predict(inputs):
        return [np.zeros(test_env.action_space.shape), 0]

def env_creator():
    env = CursorImageEnv()
    env.competitor = dummy_func
    return env

vec_env = DummyVecEnv([env_creator])
model = PPO("MlpPolicy", vec_env, n_steps=32, batch_size=8, n_epochs=1, verbose=2)
competitor_update_frequency = 32 # every 2 steps
num_of_interations = 10000

rewards = []
dice_scores = []
accumulated_predictions_list = []

for iteration in range(num_of_interations):
    print("Iteration:", iteration)
    model.learn(competitor_update_frequency, progress_bar=False)
    vec_env.env_method("set_competitor", model)

    # Collect rewards
    reward = vec_env.get_attr("reward_to_output")
    rewards.append(reward)

    # Compute final dice score at the end of the episode
    final_dice_score = vec_env.env_method("compute_final_dice_score")
    dice_scores.append(final_dice_score)

    # Collect predictions
    predictions = vec_env.get_attr("accumulated_predictions")[0]
    accumulated_predictions_list.append(predictions)
    if iteration % 20 == 0: 
        print("Dice score", final_dice_score)
        print("Reward", reward)


# Plot the rewards
plt.plot(rewards)
plt.xlabel('Iteration')
plt.ylabel('Reward')
plt.title('Reward over Iterations')
plt.show()

# Plot the dice scores
plt.plot(dice_scores)
plt.xlabel('Iteration')
plt.ylabel('Dice Score')
plt.title('Dice Score over Iterations')
plt.show()

# Plot the final accumulated predictions
final_predictions = accumulated_predictions_list[-1]
final_predictions_array = np.array(final_predictions)

x, y, z = np.where(final_predictions_array == 1)

fig = go.Figure(data=[go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=2,
        color=final_predictions_array[x, y, z],  # Color by prediction value
        colorscale='Hot',
        opacity=0.8
    )
)])

fig.update_layout(title='3D Visualization of Final Predictions')
fig.show()

# Check the number of voxels with value 1
num_voxels = np.sum(final_predictions_array == 1)
print("Number of voxels with cancer:", num_voxels)

# Reward summary
reward_average = np.mean(rewards)
reward_std = np.std(rewards)
print("Average of rewards", reward_average)
print("Standard deviation of rewards", reward_std)

# Dice score summary
dice_score_avg = np.mean(dice_scores)
dice_score_std = np.std(dice_scores)
print("Average of dice score", dice_score_avg)
print("Standard deviation of dice score", dice_score_std)