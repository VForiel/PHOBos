import time
import json
import yaml
from .. import SANDBOX_MODE

# Conditional import based on mode
if SANDBOX_MODE:
    from ..sandbox.serial_mock import Serial as serial_Serial
else:
    import serial
    serial_Serial = serial.Serial

#==============================================================================
# Pupil Mask Class
#==============================================================================

class PupilMask():
    """
    Class to control the mask wheel in the optical system.
    
    ⚠️ It is highly recommended to reset the mask to the home position using the `reset=True` parameter when initalizing the PupilMask object.

    Attributes
    ----------
    zaber_h : Zaber
        Instance of the Zaber class for controlling the horizontal motor.
    zaber_v : Zaber
        Instance of the Zaber class for controlling the vertical motor.
    newport : Newport
        Instance of the Newport class for controlling the mask wheel.
    zaber_h_home : int
        Home position for the horizontal motor (in steps).
    zaber_v_home : int
        Home position for the vertical motor (in steps).
    newport_home : float
        Angular home position for the first mask (in degrees).
    """

    def __init__(
            self,
            # On which ports the components are connected
            zaber_port:str = "/dev/ttyUSB2",
            newport_port:str = "/dev/ttyUSB1",
            zaber_h_home:int = 188490, # Horizontal axis home position (steps)
            zaber_v_home:int = 154402, # Vertical axis home position (steps)
            newport_home:float = 56.15, # Angle of the pupil mask n°1 (degree)
            reset = False, # Reset the mask to the home position
            ):
        """
        Initialize the PupilMask class.

        ⚠️ It is highly recommended to reset the mask to the home position using the `reset=True` parameter.

        Parameters
        ----------
        zaber_port : str
            Port for the Zaber motors (default is "/dev/ttyUSB0").
        newport_port : str
            Port for the Newport motor (default is "/dev/ttyUSB1").
        zaber_h_home : int
            Home position for the horizontal motor (default is 188490).
        zaber_v_home : int
            Home position for the vertical motor (default is 154402).
        newport_home : float
            Angular home position for the first mask (default is 56.15).
        reset : bool, optional
            If True, reset the mask to the home position on initialization. Default is False.
        """
        
        # Initialize the serial connections for Zaber and Newport
        zaber_session = serial_Serial(zaber_port, 115200, timeout=0.1)
        newport_session = serial_Serial(newport_port, 921600, timeout=0.1)

        self.zaber_h_home = zaber_h_home
        self.zaber_v_home = zaber_v_home
        self.newport_home = newport_home

        # Initialize the Zaber and Newport objects
        self.zaber_v = Zaber(zaber_session, 1)
        self.zaber_h = Zaber(zaber_session, 2)
        self.newport = Newport(newport_session)

        if SANDBOX_MODE:
            print(f"⛱️ [SANDBOX] PupilMask initialized - Zaber:{zaber_port}, Newport:{newport_port}")

        if reset:
            self.reset()

    #--------------------------------------------------------------------------

    def move_h(self, pos:int, abs:bool=False) -> str:
        """
        Move the mask to the horizontal by a certain number of steps.

        Parameters
        ----------
        pos : int
            Number of steps to move.
        abs : bool, optional
            If True, move to an absolute position. Default is False.

        Returns
        -------
        str
            Response from the motor after moving to the target position.
        """
        if SANDBOX_MODE:
            move_type = "absolute" if abs else "relative"
            print(f"⛱️ [SANDBOX] PupilMask - Move horizontal {move_type}: {pos} steps")
        
        if abs:
            return self.zaber_h.move_abs(pos)
        else:
            return self.zaber_h.move_rel(pos)
        
    #--------------------------------------------------------------------------

    def move_v(self, pos:int, abs:bool=False) -> str:
        """
        Move the mask vertically by a certain number of steps.

        Parameters
        ----------
        pos : int
            Number of steps to move.
        abs : bool, optional
            If True, move to an absolute position. Default is False.

        Returns
        -------
        str
            Response from the motor after moving to the target position.
        """
        if SANDBOX_MODE:
            move_type = "absolute" if abs else "relative"
            print(f"⛱️ [SANDBOX] PupilMask - Move vertical {move_type}: {pos} steps")
        
        if abs:
            return self.zaber_v.move_abs(pos)
        else:
            return self.zaber_v.move_rel(pos)
        
    #--------------------------------------------------------------------------

    def rotate_clockwise(self, pos:float, abs:bool=False) -> str:
        """
        Rotate the mask clockwise by a certain number of degrees.
        Alias: rotate()

        Parameters
        ----------
        pos : float
            Number of degrees to rotate.
        abs : bool, optional
            If True, rotate to an absolute position. Default is False.

        Returns
        -------
        str
            Response from the motor after moving to the target position.
        """
        if SANDBOX_MODE:
            move_type = "absolute" if abs else "relative"
            print(f"⛱️ [SANDBOX] PupilMask - Rotate {move_type}: {pos} degrees")
        
        if abs:
            return self.newport.move_abs(pos)
        else:
            return self.newport.move_rel(pos)
      
    def rotate(self, pos:float, abs:bool=False) -> str:
        return self.rotate_clockwise(pos, abs)

    # Apply Mask --------------------------------------------------------------

    def apply_mask(self, key:str, config_path:str=None):
        """
        Rotate the mask wheel and move the Zabers to the desired mask position.
        It can load the positions of the wheel and the zabers from a YAML file.
        In this case, `key` is the string of the key of the YAML file of the desired configuration to set.
        
        If no such file is given, `key` (string or int) is the number of the mask to put.
        The zabers remains are not moved.
        
        Parameters
        ----------
        key : str or int
            Key of the config to load.
        config_path: str, optional
            YAML file in which are stored the motors positions. See "Configuration file" documentation.
            for each wheel position. The default is None.
        
        Returns
        -------
        str
            Response from the motor after moving to the target position.
        """
        if SANDBOX_MODE:
            print(f"⛱️ [SANDBOX] PupilMask - Applying mask {key}")
        
        if config_path:
            with open(config_path, 'r') as f:
                data = json.load(f)
        
            mask_data = data['mask']['slots'][str(key)]

            wh = mask_data['a']
            zab_h = mask_data['x']
            zab_v = mask_data['y']
        
            if zab_h >= 0:
                self.zaber_h.move_abs(zab_h)
        
            if zab_v >= 0:
                self.zaber_v.move_abs(zab_v)
            
            self.newport.move_abs(wh) # Move to the desired mask position

        else:
            if SANDBOX_MODE:
                print(f"⛱️ [SANDBOX] PupilMask - No config file provided, only rotating wheel")
            mask = int(key)
            if 1 <= mask <= 6:
                self.newport.move_abs(self.newport_home + (mask-1)*60) # Move to the desired mask position
            else:
                raise ValueError("Mask index must be between 1 and 6.")
        
    #--------------------------------------------------------------------------
        
    def get_pos(self):
        """
        Get the current position of the mask.

        Returns
        -------
        float
            Current angular position of the mask wheel (in degrees).
        int
            Current position of the horizontal Zaber motor (in steps).
        int
            Current position of the vertical Zaber motor (in steps).
        """
        wheel = float(self.newport.get())
        zab2 = self.zaber_h.get()
        zab1 = self.zaber_v.get()
        
        zab1 = int(zab1.split(' ')[-1][:-2])
        zab2 = int(zab2.split(' ')[-1][:-2])
        
        return wheel, zab1, zab2
    
    def save_pos(self, key:str, config_path:str):
        """
        Save position of the wheel and the two zabers into a json file.

        Parameters
        ----------
        key : str
            Key at which saving the configuration.
        config_path : str
            Name of the json file.

        """
                
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        wh, zab_h, zab_v = self.get_pos()

        config[key] = {
            'a': wh,
            'x': zab_h,
            'y': zab_v 
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

    #--------------------------------------------------------------------------

    def reset(self) -> None:
        """
        Reset the mask wheel to the 4 vertical holes and the Zaber motors to their home positions.
        """
        if SANDBOX_MODE:
            print("⛱️ [SANDBOX] PupilMask - Resetting to home position")
        
        self.newport.home_search()
        self.apply_mask(4)
        self.zaber_h.move_abs(self.zaber_h_home)
        self.zaber_v.move_abs(self.zaber_v_home)
    
#==============================================================================
# Zaber Class
#==============================================================================

class Zaber():
    """
    Class to control the Zaber motors (axis).

    Attributes
    ----------
    id : int
        ID of the Zaber motor.
    """

    def __init__(self, session, id):
        """
        Parameters
        ----------
        session : serial.Serial
            Serial connection to the Zaber motor.
        id : int
            ID of the Zaber motor.
        """
        self._session = session
        self._id = id

    # Properties --------------------------------------------------------------

    @property
    def id(self) -> int:
        return self._id
    
    @id.setter
    def id(self, id:int) -> None:
        raise ValueError("ID cannot be changed after initialization.")
    

    # Wait --------------------------------------------------------------------

    def wait(self) -> None:
        """
        Wait for the motor to reach the target position.
        """
        if SANDBOX_MODE:
            print(f"⛱️ [SANDBOX] Zaber motor {self.id} - Waiting for movement...")
            time.sleep(0.1)  # Reduced wait simulation
            print(f"⛱️ [SANDBOX] Zaber motor {self.id} - Position reached")
        else:
            position = None
            while position != self.get():
                position = self.get()
                time.sleep(0.1)

    #--------------------------------------------------------------------------

    def send_command(self, command):
        """
        Send a command to the motor and return the response.

        Parameters
        ----------
        command : str
            Command to send to the motor.

        Returns
        -------
        str
            Response from the motor.
        """
        if SANDBOX_MODE:
            print(f"⛱️ [SANDBOX] Zaber motor {self.id} - Command: {command}")
        
        self._session.write(f"/{self.id} {command}\r\n".encode())
        return self._session.readline().decode()
    
    #--------------------------------------------------------------------------

    def get(self) -> int:
        """
        Get the current position of the motor.

        Returns
        -------
        int
            Current position of the motor (in steps).
        """
        return self.send_command("get pos")
    
    #--------------------------------------------------------------------------
    
    def move_abs(self, pos:int) -> str:
        """
        Move the motor to an absolute position.

        Parameters
        ----------
        pos : int
            Target position in steps.

        Returns
        -------
        str
            Response from the motor after moving to the target position.
        """
        response = self.send_command(f"move abs {pos}")
        self.wait()
        return response
    
    #--------------------------------------------------------------------------

    def move_rel(self, pos:int) -> str:
        """
        Move the motor by a relative number of steps.

        Parameters
        ----------
        pos : int
            Number of steps to move.

        Returns
        -------
        str
            Response from the motor after moving to the target position.
        """
        response = self.send_command(f"move rel {pos}")
        self.wait()
        return response
    
#==============================================================================
# Newport Class
#==============================================================================

class Newport():
    """
    Class to control the Newport motor (wheel).
    
    ⚠️ If the command sent to the Newport motor doesn't work but no error is raised, ensure the Newport know it's home position by running the `home_search()` method first.
    """

    def __init__(self, session):
        """
        Initialize the Newport motor.

        Parameters
        ----------
        session : serial.Serial
            Serial connection to the Newport motor.
        """
        self._session = session

    #--------------------------------------------------------------------------

    def home_search(self) -> str:
        """
        Move the motor to the home position.

        Returns
        -------
        str
            Response from the motor after moving to home position.
        """
        if SANDBOX_MODE:
            print("⛱️ [SANDBOX] Newport - Home search initiated")
        
        response = self.send_command("1OR?")
        self.wait()
        return response

    # Wait --------------------------------------------------------------------

    def wait(self) -> None:
        """
        Wait for the motor to reach the target position.
        """
        if SANDBOX_MODE:
            print("⛱️ [SANDBOX] Newport - Waiting for movement...")
            time.sleep(0.1)  # Reduced wait simulation
            print("⛱️ [SANDBOX] Newport - Position reached")
        else:
            position = None
            while position != self.get():
                position = self.get()
                time.sleep(0.1)

    #--------------------------------------------------------------------------

    def send_command(self, command):
        """
        Send a command to the motor and return the response.

        Parameters
        ----------
        command : str
            Command to send to the motor.

        Returns
        -------
        str
            Response from the motor.
        """
        if SANDBOX_MODE:
            print(f"⛱️ [SANDBOX] Newport - Command: {command}")

        self._session.write(f"{command}\r\n".encode())
        return self._session.readline().decode()
    
    #--------------------------------------------------------------------------

    def get(self) -> float:
        """
        Get the current angular position of the motor (in degrees).

        Returns
        -------
        float
            Current angular position (in degrees) of the motor in degrees.
        """
        return float(self.send_command("1TP?")[3:-2])
    
    #--------------------------------------------------------------------------

    def move_abs(self, pos:float) -> str:
        """
        Rotate the motor to an absolute angular position (in degrees).

        Parameters
        ----------
        pos : int
            Target angular position in degrees.

        Returns
        -------
        str
            Response from the motor after moving to the target position.
        """
        response = self.send_command(f"1PA{pos}")
        self.wait()
        return response
    
    #--------------------------------------------------------------------------

    def move_rel(self, pos:int) -> str:
        """
        Rotate the motor by a relative angle.

        Parameters
        ----------
        pos : int
            Angle to rotate in degrees.

        Returns
        -------
        str
            Response from the motor after moving to the target position.
        """
        response = self.send_command(f"1PR{pos}")
        self.wait()
        return response