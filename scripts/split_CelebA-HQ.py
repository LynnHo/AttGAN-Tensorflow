import os
import random

# ==============================================================================
# =                                   param                                    =
# ==============================================================================

label_file = './data/CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt'
save_dir = './data/CelebAMask-HQ'


# ==============================================================================
# =                                    run                                     =
# ==============================================================================

with open(label_file, 'r') as f:
    lines = f.readlines()[2:]

random.seed(100)
random.shuffle(lines)

lines_train = lines[:26500]
lines_val = lines[26500:27000]
lines_test = lines[27000:]

with open(os.path.join(save_dir, 'train_label.txt'), 'w') as f:
    f.writelines(lines_train)

with open(os.path.join(save_dir, 'val_label.txt'), 'w') as f:
    f.writelines(lines_val)

with open(os.path.join(save_dir, 'test_label.txt'), 'w') as f:
    f.writelines(lines_test)
