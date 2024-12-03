from gops.env.env_gen_ocp.resources.idsim_model.utils.vedio_utils.generate_gif import process_batch
from gops.env.env_gen_ocp.resources.idsim_model.utils.vedio_utils.generate_vedio import \
    gen_video_cv2_mp4, gen_video_cv2_avi, gen_video_imio_mp4

path = "/home/idlab/code/qx-oracle/data_qx/draw/DSACTPI_241201-183503/12-02-16:37:30"

fname_avi = "regen.avi"
fname_mp4 = "regen.mp4"

# gen_video_cv2_avi(path, fname_avi)
# gen_video_imio_mp4(path, fname_mp4)
process_batch(path, "res")
