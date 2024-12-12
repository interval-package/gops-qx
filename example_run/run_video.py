import os
from gops.env.env_gen_ocp.resources.idsim_model.utils.vedio_utils.generate_gif import process_batch
from gops.env.env_gen_ocp.resources.idsim_model.utils.vedio_utils.generate_vedio import \
    gen_video_cv2_mp4, gen_video_cv2_avi, gen_video_imio_mp4

path = "/home/idlab/code/qx-oracle/data_qx/train/12-09-16:47:45"

fname_avi = "regen.avi"
fname_mp4 = "regen.mp4"

# gen_video_cv2_avi(path, fname_avi)
gen_video_imio_mp4(path, os.path.join(path, fname_mp4))
# process_batch(path, "res")
