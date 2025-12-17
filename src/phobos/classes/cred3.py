import numpy as np
from .. import shm, SANDBOX_MODE


class Cred3:
    """
    Class to interface with the Cred3 camera via shared memory.
    
    The camera writes frames to a shared memory location that can be read
    by this class. Optionally, dark frames can be subtracted.
    
    Parameters
    ----------
    img_shm_path : str, optional
        Shared memory path for the camera frames. Default is '/dev/shm/cred1.im.shm'.
    dark_shm_path : str, optional
        Shared memory path for the dark frame. Default is '/dev/shm/cred3_dark.im.shm'.
    semid : int, optional
        Semaphore ID for frame synchronization. Default is 0.
    use_dark : bool, optional
        Whether to subtract dark frames. Default is True.
    
    Attributes
    ----------
    cam : shm object
        Shared memory instance for camera frames.
    dark : ndarray or None
        Dark frame array, or None if use_dark is False.
    semid : int
        Semaphore ID for synchronization.
    
    Examples
    --------
    >>> camera = Cred3()
    >>> img = camera.get_image()
    >>> outputs = camera.get_outputs()
    """
    
    def __init__(self, 
                 img_shm_path: str = '/dev/shm/cred1.im.shm',
                 dark_shm_path: str = '/dev/shm/cred3_dark.im.shm',
                 semid: int = 0,
                 use_dark: bool = True):
        """
        Initialize the Cred3 camera interface.
        """
        if SANDBOX_MODE:
            print("⛱️ [SANDBOX] Cred3 running in mock mode")
        
        self.img_shm_path = img_shm_path
        self.dark_shm_path = dark_shm_path
        self.semid = semid
        self.use_dark = use_dark
        
        # Initialize shared memory for camera
        self.cam = shm(img_shm_path, nosem=False)
        self.cam.catch_up_with_sem(semid)
        
        # Initialize dark frame if needed
        self.dark_shm_obj = None
        if use_dark:
            try:
                self.dark_shm_obj = shm(dark_shm_path)
                self.dark = self.dark_shm_obj.get_latest_data()
                mode_prefix = "⛱️ [SANDBOX] " if SANDBOX_MODE else ""
                print(f"{mode_prefix}Cred3 camera initialized with dark subtraction")
            except Exception as e:
                print(f"⚠️ Could not load dark frame: {e}")
                self.dark = None
        else:
            self.dark = None
            mode_prefix = "⛱️ [SANDBOX] " if SANDBOX_MODE else ""
            print(f"{mode_prefix}Cred3 camera initialized without dark subtraction")
    
    def get_image(self, subtract_dark: bool = True) -> np.ndarray:
        """
        Get the latest image from the shared memory.
        
        Parameters
        ----------
        subtract_dark : bool, optional
            If True, subtract the dark frame. If None, uses the default
            set during initialization. Default is True.
        
        Returns
        -------
        img : ndarray
            Latest camera frame, optionally dark-subtracted.
        
        Examples
        --------
        >>> camera = Cred3()
        >>> img = camera.get_image()
        >>> img_no_dark = camera.get_image(subtract_dark=False)
        """
        img = self.cam.get_latest_data(self.semid)
        
        # Determine whether to subtract dark
        if subtract_dark is None:
            subtract_dark = self.use_dark
        
        if subtract_dark:
            # Update dark frame if possible
            if self.dark_shm_obj is not None:
                try:
                    self.dark = self.dark_shm_obj.get_latest_data()
                except Exception:
                    pass  # Keep using existing dark if update fails
            
            if self.dark is not None:
                img = img - self.dark
        
        return img
    
    def crop_outputs_from_image(self, 
                               img: np.ndarray, 
                               crop_centers: np.ndarray, 
                               crop_sizes=10) -> list[np.ndarray]:
        """
        Crop regions around specified centers from an image and return them.
        
        Parameters
        ----------
        img : ndarray
            Input image to crop from.
        crop_centers : ndarray
            Array of (x, y) coordinates for crop centers, shape (N, 2).
        crop_sizes : int or tuple, optional
            Size of the crop window. If int, a square window of this size
            is used for all outputs. If tuple of length N, each output
            gets its own crop size. Default is 10 pixels.
            
        Returns
        -------
        crops : list of ndarray
            List of cropped sub-images.
        """

        img = np.transpose(img)  # Transpose to match (x, y) indexing
        
        # Handle crop_sizes - convert to array
        n_outputs = crop_centers.shape[0]
        if isinstance(crop_sizes, (int, float)):
            crop_sizes_array = np.full(n_outputs, int(crop_sizes))
        else:
            crop_sizes_array = np.array(crop_sizes)
            if len(crop_sizes_array) != n_outputs:
                raise ValueError(f"crop_sizes length ({len(crop_sizes_array)}) must match "
                               f"number of centers ({n_outputs})")
        
        # Compute flux for each output
        crops = []
        for i in range(n_outputs):
            x_center, y_center = crop_centers[i]
            crop_size = crop_sizes_array[i]
            half_size = crop_size // 2
            
            # Define crop boundaries
            x1 = int(x_center - half_size)
            x2 = int(x1 + crop_size)
            y1 = int(y_center - half_size)
            y2 = int(y1 + crop_size)
            
            # Extract crop
            # Handle boundary conditions to avoid errors if crop is outside image
            try:
                crop = img[x1:x2, y1:y2]
                crops.append(crop.T)
            except IndexError:
                print(f"⚠️ Crop region {i} outside image boundaries")
                # Append empty array or zeros matching size if possible, or None
                # For robustness let's append a zero array of expected size
                crops.append(np.zeros((crop_size, crop_size)))
        
        return [crop.copy() for crop in crops]

    def get_outputs(self,
                   crop_centers: np.ndarray,
                   crop_sizes=10,
                   subtract_dark: bool = True,
                   flux_mode: str = 'mean') -> np.ndarray:
        """
        Get the flux around specified output centers.
        
        This method crops regions around specified centers and returns
        the flux in each region. The flux can be the mean pixel value
        or the sum of pixel values.
        
        Parameters
        ----------
        crop_centers : ndarray
            Array of (x, y) coordinates for crop centers, shape (N, 2).
        crop_sizes : int or tuple, optional
            Size of the crop window. If int, a square window of this size
            is used for all outputs. If tuple of length N, each output
            gets its own crop size. Default is 10 pixels.
        subtract_dark : bool, optional
            Whether to subtract dark frame. If None, uses initialization
            default. Default is True.
        flux_mode : str, optional, 'mean' or 'sum'
            Method to compute the flux. 
            'mean': average of pixel values (default)
            'sum': sum of pixel values
        
        Returns
        -------
        flux : ndarray
            Flux in each cropped region, shape (N,).
        
        Examples
        --------
        >>> camera = Cred3()
        >>> # Get averaged outputs with default centers
        >>> flux = camera.get_outputs()
        >>> 
        >>> # Get integrated flux (sum)
        >>> flux_sum = camera.get_outputs(flux_mode='sum')
        >>> 
        >>> # Custom centers and crop size
        >>> centers = np.array([(100, 200), (300, 400)])
        >>> flux = camera.get_outputs(crop_centers=centers, crop_sizes=20)
        """
        # Get the latest image
        img = self.get_image(subtract_dark=subtract_dark)
        
        crops = self.crop_outputs_from_image(img, crop_centers=crop_centers, crop_sizes=crop_sizes)
        
        # Compute flux
        flux = np.zeros(len(crops))
        for i, crop in enumerate(crops):        
            if flux_mode == 'sum':
                flux[i] = np.sum(crop)
            elif flux_mode == 'mean':
                flux[i] = np.mean(crop)
            else:
                raise ValueError(f"Unknown flux_mode: {flux_mode}. Use 'mean' or 'sum'.")
                
        return flux
    
    def update_dark(self, dark_shm_path: str = None):
        """
        Update the dark frame from shared memory.
        
        Parameters
        ----------
        dark_shm_path : str, optional
            Shared memory path for the new dark frame. If None, uses
            the default dark_shm_path. Default is None.
        
        Examples
        --------
        >>> camera = Cred3()
        >>> camera.update_dark()  # Reload dark from default location
        """
        if dark_shm_path is None:
            dark_shm_path = self.dark_shm_path
        
        try:
            dk = shm(dark_shm_path)
            self.dark = dk.get_latest_data()
            self.use_dark = True
            print(f"Dark frame updated from {dark_shm_path}")
        except Exception as e:
            print(f"⚠️ Could not update dark frame: {e}")
    
    def take_darks(self, nb_frames: int = 1000, save_to_shm: bool = True) -> np.ndarray:
        """
        Acquire dark frames and compute the average.
        
        This method acquires a specified number of frames, computes their average,
        and optionally saves the result to shared memory for use as a dark frame.
        
        Parameters
        ----------
        nb_frames : int, optional
            Number of frames to acquire and average. Default is 1000.
        save_to_shm : bool, optional
            If True, save the averaged dark frame to shared memory at
            self.dark_shm_path. Default is True.
        
        Returns
        -------
        dark_mean : ndarray
            The averaged dark frame.
        
        Notes
        -----
        - Ensure the camera source is turned OFF before acquiring darks
        - The method checks for late frames (when semaphore value > 0)
        - The dark frame is automatically loaded if save_to_shm is True
        
        Examples
        --------
        >>> camera = Cred3()
        >>> # Acquire 100 dark frames (default)
        >>> dark = camera.take_darks()
        >>> 
        >>> # Acquire 50 frames without saving to shared memory
        >>> dark = camera.take_darks(nb_frames=50, save_to_shm=False)
        """
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False
            print("⚠️ tqdm not available, progress bar disabled")
        
        if SANDBOX_MODE:
            print(f"⛱️ [SANDBOX] Simulating acquisition of {nb_frames} dark frames")
            # In sandbox mode, generate random dark frames
            import time
            time.sleep(0.5)  # Simulate acquisition time
            dark_mean = np.random.randint(0, 100, size=(320, 256), dtype=np.uint16)
            print(f"⛱️ [SANDBOX] Dark acquisition complete")
            return dark_mean
        
        # images = []
        log_sem = []
        
        print(f"Taking {nb_frames} dark frames...")
        
        # Catch up with semaphore before starting
        self.cam.catch_up_with_sem(self.semid)
        
        # Acquire frames
        iterator = tqdm(range(nb_frames)) if use_tqdm else range(nb_frames)
        images = np.zeros_like(self.cam.get_latest_data(self.semid))
        for ii in iterator:
            img = self.cam.get_latest_data(self.semid)
            # images.append(img)
            images += img 
            semval = self.cam.sems[self.semid].value
            log_sem.append(semval)
        
        # images = np.array(images)
        print('Acquisition complete')
        
        # Check for late frames
        log_sem = np.array(log_sem)
        if np.any(log_sem > 0):
            mask = np.where(log_sem > 0)[0]
            print(f'⚠️ Total late frames: {len(mask)}')
        
        # Compute average dark
        # dark_mean = images.mean(axis=0)
        dark_mean = images / nb_frames
        
        # Save to shared memory if requested
        if save_to_shm:
            print(f'Saving dark frame to {self.dark_shm_path}...')
            try:
                dark_shm_obj = shm(self.dark_shm_path, data=dark_mean)
                print('✅ Dark frame saved to shared memory')
                
                # Update the internal dark frame
                self.dark = dark_mean
                self.dark_shm_obj = dark_shm_obj
                self.use_dark = True
            except Exception as e:
                print(f"⚠️ Could not save dark to shared memory: {e}")
        
        return dark_mean
    
    def close(self):
        """
        Close the shared memory connections.
        
        This method is provided for compatibility but shared memory
        objects are typically managed by the system.
        """
        # Shared memory objects don't need explicit closing in xaosim
        print("Cred3 camera interface closed")
