import json
import numpy as np
import os
import matplotlib.pyplot as plt

train_folder = 'brats_fold0//session1_2'
dice_path = os.path.join('..', 'models', train_folder, 'dice.json')

with open(dice_path, 'r') as f:
    dice_dict = json.load(f)

dice_vals = np.array(dice_dict['dice_values'])

four_classes = dice_vals[:, 2:]
mean_dice = np.mean(four_classes, axis=0)
std_dice = np.std(four_classes, axis=0)
print mean_dice
print std_dice

colors = ['lightgreen', 'yellow', 'orange', 'lightcoral']
bplot = plt.boxplot(four_classes, patch_artist=True, showmeans=True)
plt.yticks(np.arange(0., 1., 0.1))
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)
ax = plt.gca()
ax.grid(True)
plt.show()