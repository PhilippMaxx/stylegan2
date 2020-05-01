from pathlib import Path
import sys
import glob
import os
from tqdm import tqdm
import random
import argparse
from datetime import datetime
from PIL import Image

# StyleGAN Utils
from stylegan_utils import load_network, create_video

import dnnlib
import dataset_tool
import run_projector
import projector
import training.dataset
import training.misc

#----------------------------------------------------------------------------

def setup_network(network: str):
    """load network from pkl"""
    Gs, _, _ = load_network(network)
    img_size = Gs.output_shape[2:]
    return Gs, img_size

#----------------------------------------------------------------------------

def crop (img, size):
    """crop img central and resize to size"""
    w, h = img.size   # Get dimensions
    mx = min(w, h)

    left = (w - mx)/2
    top = (h - mx)/2
    right = (w + mx)/2
    bottom = (h + mx)/2

    img_crop = img.crop((left, top, right, bottom))
    return img_crop.resize(size, resample=Image.BILINEAR)

#----------------------------------------------------------------------------

def adjust(image_dir, size, num_images):
    """adjusts images to the network"""
    img_paths = glob.glob(image_dir+'/*.jpg')
    if not len(img_paths) <= num_images:
        img_paths = random.sample(img_paths, num_images)
    return [crop(Image.open(img_path), size) for img_path in img_paths]

#----------------------------------------------------------------------------

def project_images(Gs, images_dir, tfrecord_dir, data_dir, num_snapshots, pure_projector=False):
    """setup projector"""
    print('Setting up projector')
    proj = projector.Projector()
    proj.set_network(Gs)

    # generate tfrecords
    nb_images = dataset_tool.create_from_images(str(tfrecord_dir), str(images_dir), True)

    # loading images from tfrecords
    dataset_obj = training.dataset.load_dataset(tfrecord_dir=tfrecord_dir,
                                                max_label_size=0, verbose=True, repeat=False, shuffle_mb=0)
    assert dataset_obj.shape == Gs.output_shape[1:]

    # project all loaded images
    print('=======================')
    for image_idx in tqdm(range(nb_images)):
        print(f'Projecting image {image_idx + 1}/{nb_images}')

        images, _labels = dataset_obj.get_minibatch_np(1)
        images = training.misc.adjust_dynamic_range(images, [0, 255], [-1, 1])

        run_path = data_dir / f'out_{image_idx}'
        run_path.mkdir()
        run_projector.project_image(proj, targets=images,
                                    png_prefix=dnnlib.make_run_dir_path(str(run_path / 'image_')),
                                    num_snapshots=num_snapshots)

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=
                                     '''StyleGAN2 projector2.
                                     Run 'python %(prog)s <subcommand> --help' for subcommand help.''',
                                     formatter_class=argparse.RawDescriptionHelpFormatter
                                     )
    parser.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    parser.add_argument('--image-dir', help='Image directory', required=True)
    parser.add_argument('--num-snapshots', type=int, help='Number of snapshots (default: %(default)s)', default=100)
    parser.add_argument('--num-images', type=int, help='Number of images to project (default: %(default)s)', default=5)
    parser.add_argument('--projection-dir', help='location for projections',
                        default='projection', metavar='DIR')

    args = parser.parse_args()

    Gs, img_size = setup_network(args.network_pkl)
    images = adjust(args.image_dir, img_size, args.num_images)
    tmp = Path(args.image_dir) / 'tmp'
    if not tmp.is_dir():
        tmp.mkdir()

    images = [image.save(tmp / f'{e}.jpg') for e, image in enumerate(images)]

    project = Path(args.projection_dir) / str(datetime.now().strftime("%Y%m%d_%H%M%S"))
    if not project.is_dir():
        project.mkdir(parents=True)

    project_images(Gs, images_dir=tmp, tfrecord_dir=project / 'tfrecords', data_dir=project,
                   num_snapshots=args.num_snapshots, pure_projector=True)

    for num in range(len(images)):
        create_video(project / f'out_{num}', Path(args.projection_dir) / 'out_{}_{}.mp4'.format(num, datetime.now().strftime("%Y%m%d_%H%M%S")), fps=10)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
