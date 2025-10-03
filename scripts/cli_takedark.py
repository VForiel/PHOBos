#!/usr/bin/env python

import argparse
import numpy as np
from xaosim.shmlib import shm
from tqdm import tqdm
# import itertools

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script to acquire signal-less frames and create an average frame to perform cosmetics on data cubes. \
        Turn off the source then launch the script.")
    
    parser.add_argument("-n", "--nb_frames", required=True, type=int, help="Number of frames to acquire.")
    args = parser.parse_args()
    nbframes = int(args.nb_frames)
    
    # nbframes = 800 # number of frames to average
    
    cam = shm('/dev/shm/cred1.im.shm', nosem=False) # the source of data
    
    images = []
    print("Taking Darks...")
    
    semid = 0
    cam.catch_up_with_sem(semid)
    log_sem = []
    
    for ii in tqdm(range(nbframes)):
        img = cam.get_latest_data(semid)
        images.append(img)
        semval = cam.sems[semid].value
        log_sem.append(semval)
            
    images = np.array(images)
    print('Done')
    
    log_sem = np.array(log_sem)
    if np.any(log_sem > 0):
        mask = np.where(log_sem > 0)[0]
        for idx in mask:
            print('Late frame', idx, log_sem[idx])
        print('Total', len(mask))
    
    # pairs = list(itertools.combinations(range(nbframes), 2))
    
    # for i, j in pairs:
    #     if np.array_equiv(images[i], images[j]):
    #         print('Same frame', i, j)
    
    dark_mean = images.mean(0)
    
    print('Save Dark...')
    dark = shm('/dev/shm/cred3_dark.im.shm', data=dark_mean)
    print('Done')