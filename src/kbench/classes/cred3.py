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
        if use_dark:
            try:
                dk = shm(dark_shm_path)
                self.dark = dk.get_latest_data()
                mode_prefix = "⛱️ [SANDBOX] " if SANDBOX_MODE else ""
                print(f"{mode_prefix}Cred3 camera initialized with dark subtraction")
            except Exception as e:
                print(f"⚠️ Could not load dark frame: {e}")
                self.dark = None
        else:
            self.dark = None
            mode_prefix = "⛱️ [SANDBOX] " if SANDBOX_MODE else ""
            print(f"{mode_prefix}Cred3 camera initialized without dark subtraction")
    
    def get_image(self, subtract_dark: bool = None) -> np.ndarray:
        """
        Get the latest image from the shared memory.
        
        Parameters
        ----------
        subtract_dark : bool, optional
            If True, subtract the dark frame. If None, uses the default
            set during initialization. Default is None.
        
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
        
        if subtract_dark and self.dark is not None:
            img = img - self.dark
        
        return img
    
    def get_outputs(self, 
                   crop_centers: np.ndarray = None,
                   crop_sizes: int | tuple = 10,
                   subtract_dark: bool = None) -> np.ndarray:
        """
        Get the mean intensity around specified output centers.
        
        This method crops regions around specified centers and returns
        the mean flux in each region.
        
        Parameters
        ----------
        crop_centers : ndarray, optional
            Array of (x, y) coordinates for crop centers, shape (N, 2).
            Default centers correspond to the 4 main outputs:
            [(594, 114), (499, 90), (404, 66), (309, 42)]
        crop_sizes : int or tuple, optional
            Size of the crop window. If int, a square window of this size
            is used for all outputs. If tuple of length N, each output
            gets its own crop size. Default is 10 pixels.
        subtract_dark : bool, optional
            Whether to subtract dark frame. If None, uses initialization
            default. Default is None.
        
        Returns
        -------
        flux : ndarray
            Mean intensity in each cropped region, shape (N,).
        
        Examples
        --------
        >>> camera = Cred3()
        >>> # Get outputs with default centers
        >>> flux = camera.get_outputs()
        >>> 
        >>> # Custom centers and crop size
        >>> centers = np.array([(100, 200), (300, 400)])
        >>> flux = camera.get_outputs(crop_centers=centers, crop_sizes=20)
        >>> 
        >>> # Different crop sizes for each output
        >>> flux = camera.get_outputs(crop_sizes=(10, 15, 10, 10))
        """
        # Default crop centers (4 main outputs)
        if crop_centers is None:
            crop_centers = np.array([(594, 114),
                                    (499, 90),
                                    (404, 66),
                                    (309, 42)])
        else:
            crop_centers = np.array(crop_centers)
        
        # Get the latest image
        img = self.get_image(subtract_dark=subtract_dark)
        
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
        flux = np.zeros(n_outputs)
        for i in range(n_outputs):
            x_center, y_center = crop_centers[i]
            crop_size = crop_sizes_array[i]
            half_size = crop_size // 2
            
            # Define crop boundaries
            x1 = int(x_center - half_size)
            x2 = int(x_center + half_size + 1)
            y1 = int(y_center - half_size)
            y2 = int(y_center + half_size + 1)
            
            # Extract crop and compute mean
            crop = img[x1:x2, y1:y2]
            flux[i] = np.mean(crop)
        
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
    
    def close(self):
        """
        Close the shared memory connections.
        
        This method is provided for compatibility but shared memory
        objects are typically managed by the system.
        """
        # Shared memory objects don't need explicit closing in xaosim
        print("Cred3 camera interface closed")
