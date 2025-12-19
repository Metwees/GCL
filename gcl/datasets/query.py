import os
import os.path as osp
import glob
import re

ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]
def collect_gan_images(
    original_list,
    gan_root,
    angles=ANGLES
):
    """
    Returns:
      gan_paths[N][Q]
      gan_ids[N]
      gan_cams[N]
      gan_list_flat[(path, pid, camid)]
    """

    gan_paths = []
    gan_ids   = []
    gan_cams  = []
    gan_list_flat = []

    #for path, pid, camid in original_list:
    for i, (path, pid, camid) in enumerate(original_list):
        if i % 500 == 0:
            print(f"[GAN] Processed {i}/{len(original_list)*len(angles)} images")
        
        filename = osp.basename(path)

        img_paths = []

        for a in angles:
            p = osp.join(gan_root, str(a), filename)
            if not osp.exists(p):
                raise FileNotFoundError(f"Missing GAN image: {p}")

            img_paths.append(p)
            gan_list_flat.append((p, pid, camid))

        gan_paths.append(img_paths)
        gan_ids.append(pid)
        gan_cams.append(camid)

    return gan_paths, gan_ids, gan_cams, gan_list_flat