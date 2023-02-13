import csv
import os
from tqdm import tqdm
import pandas as pd
from argparse import Namespace
import release_renderer_kinect

OUT_DIR_IMGS = "scene_silhouette_imgs_scenes"
OUT_FILE_LABELS = "labels.csv"

def generate_screenshots(out_dir_imgs, scene_name, recording_name):
    args = Namespace(
        model_folder='models',
        model_type='smplx',
        num_pca_comps=12,
        recording_name=recording_name,
        release_data_root='dataset',
        rendering_mode='3d',
        save_root=out_dir_imgs,
        save_undistorted_img=False,
        scale=4,
        scene_name=scene_name,
        start=0,
        step=50,
        view='master'
    )
    lbls = release_renderer_kinect.main(args)
    return lbls

if __name__ == "__main__":
    # Read scene - recording pairs
    df = pd.read_csv('dataset/data_info_release.csv')
    scenes_recordings = df[['scene_name', 'recording_name']]
    # Generate img - label pairs
    os.makedirs(OUT_DIR_IMGS)
    all_lbls = {'img_name':[], 'x':[], 'y':[], 'z':[]}
    for _, row in tqdm(scenes_recordings.iterrows()):
        # print("scene_name, recording_name", row['scene_name'], row['recording_name'])
        lbls = generate_screenshots(OUT_DIR_IMGS, row['scene_name'], row['recording_name'])
        for k in all_lbls.keys():
            all_lbls[k] += lbls[k]
    pd.DataFrame.from_dict(all_lbls).to_csv(OUT_FILE_LABELS, index=False)
    print(f"Wrote labels into {OUT_FILE_LABELS}")