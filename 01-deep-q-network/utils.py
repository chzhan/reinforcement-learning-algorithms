import numpy as np
import cv2
import random 


# process the image..
def pre_processing(x):
    x = x[:, :, (2, 1, 0)]
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    x = np.float32(x) / 255
    x = cv2.resize(x, (84, 84))
        
    return x

def reward_wrapper(reward):
    return np.sign(reward)

def select_action(action, epsilon, num_actions):
    # transfer the action from the gpu to cpu
    action_selected = action.cpu().numpy()[0]
    action_selected = int(action_selected)
    # greedy...
    dice = random.uniform(0, 1)
    if dice >= 1 - epsilon:
        action_selected = random.randint(0, num_actions - 1)

    return action_selected

