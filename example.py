import dm
import flow_vis
import matplotlib.pyplot as plt

# Target and reference image paths:
tar_im_path = "MPI-Sintel/training/clean/alley_1/frame_0001.png"
ref_im_path = "MPI-Sintel/training/clean/alley_1/frame_0002.png"

# Run matching:
res = dm.dm_match_pair(tar_im_path, ref_im_path, radius=40)

# Visualize matching results:
flow_color = flow_vis.flow_to_color(res, convert_to_bgr=False)
plt.imshow(flow_color)
plt.axis('off')
plt.show()
