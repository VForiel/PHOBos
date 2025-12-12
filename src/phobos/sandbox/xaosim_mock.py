"""
Mock xaosim shared memory module for sandbox mode.
Simulates shared memory communications for camera interface.
"""

import numpy as np


class shm:
    """
    Mock of xaosim.shmlib.shm class for sandbox mode.
    Simulates shared memory interface for camera frames.
    """
    
    def __init__(self, name: str, nosem: bool = True):
        """
        Simulates creating a shared memory connection.
        
        Parameters
        ----------
        name : str
            Shared memory name (e.g., '/dev/shm/cred1.im.shm').
        nosem : bool, optional
            Whether to use semaphores. Default is True.
        """
        self.name = name
        self.nosem = nosem
        
        print(f"⛱️ [SANDBOX] Creating shared memory: {name}")
        
        # Create a dummy image (640x512 typical for Cred3)
        # Initialize with realistic camera-like data
        self._data = np.random.randint(100, 1000, (640, 512), dtype=np.uint16)
        
        # Add some structure to make it look more realistic
        # Create some "hot spots" to simulate outputs
        if 'dark' not in name:
            # Add bright spots at typical output locations
            for x, y in [(594, 114), (499, 90), (404, 66), (309, 42)]:
                if 0 <= x < 640 and 0 <= y < 512:
                    self._data[x-5:x+5, y-5:y+5] += 2000
    
    def catch_up_with_sem(self, semid: int):
        """
        Mock semaphore catch up.
        
        Parameters
        ----------
        semid : int
            Semaphore ID.
        """
        print(f"⛱️ [SANDBOX] Catching up with semaphore {semid} for {self.name}")
    
    def get_latest_data(self, semid: int = None) -> np.ndarray:
        """
        Return mock image data.
        
        Parameters
        ----------
        semid : int, optional
            Semaphore ID for synchronization.
        
        Returns
        -------
        data : ndarray
            Mock camera frame with simulated noise.
        """
        # Add some variation to simulate real camera noise
        noise = np.random.randint(-10, 10, self._data.shape, dtype=np.int16)
        result = (self._data.astype(np.int32) + noise).clip(0, 65535).astype(np.uint16)
        
        if semid is not None:
            print(f"⛱️ [SANDBOX] Getting latest data from {self.name} (semid={semid})")
        
        return result
