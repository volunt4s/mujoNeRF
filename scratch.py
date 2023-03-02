import numpy as np

img_cnt = np.arange(168)
[train_idx, test_idx, val_idx] = np.random.choice(img_cnt, [img_cnt-20, 10, 10], replace=False)

