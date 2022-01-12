from glob import glob
import pathlib
import subprocess


checkpoints = glob('checkpoints/*_epoch_10_*')

for checkpoint in checkpoints:
    checkpoint_replaced = checkpoint.replace('_epoch_10_', '_current_')
    print(checkpoint)
    print(checkpoint_replaced)
    subprocess.run([f'cp {checkpoint} {checkpoint_replaced}'],shell=True)
