import glob
from PIL import Image
import time
import os

def generate(fp_in, fp_out):
    # filepaths

    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif

    file_list = glob.glob(fp_in)

    img, *imgs = [Image.open(f) for f in sorted(file_list)]
    fps = int(len(imgs)/60)
    print(f'FPS: {fps}')
    img.save(fp=fp_out, format='GIF', append_images=imgs,
            save_all=True, duration=fps, loop=0)
    
    for del_img in file_list:
        os.remove(del_img)

    print(f'Saved to {fp_out}')

if __name__ == '__main__':
    fp_in = "out/residual/*.png"
    fp_out = f"out/residual_{int(time.time())}.gif"
    generate(fp_in, fp_out)
    
    fp_in = "out/hidden/*.png"
    fp_out = f"out/hidden_{int(time.time())}.gif"
    generate(fp_in, fp_out)