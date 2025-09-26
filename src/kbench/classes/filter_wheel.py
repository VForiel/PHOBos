import time
from .. import SANDBOX_MODE

# Conditional import based on mode
if SANDBOX_MODE:
    from ..sandbox.serial_mock import Serial as serial_Serial
else:
    import serial
    serial_Serial = serial.Serial

class FilterWheel():
    def __init__(self, filter_port:str = "/dev/ttyUSB2"):
        """
        Class to control the Thorlabs filter wheel. The wheel has 6 positions:
            - 1: ND?
            - 2: ND?
            - 3: ND?
            - 4: ND?
            - 5: ND?
            - 6: ND?

        Parameters
        ----------
        filter_port : str, optional
            Serial port number. The default is "/dev/ttyUSB2".

        Returns
        -------
        None.

        """
      
        self.session = serial_Serial(filter_port, 115200, timeout=0.1)
        
        if SANDBOX_MODE:
            print(f"⛱️ [SANDBOX] Filter Wheel initialized on port {filter_port}")
        else:
            print(f"Filter Wheel connected on port {filter_port}")
        
    def _purge(self):
        """
        Purge all the history of the responses of the filter wheel.
        """
        # Reading the lines actually flush the info after the request
        dummy = self.session.readlines()
    
    def close(self):
        """
        Close the serial connection.
        """
        if SANDBOX_MODE:
            print("⛱️ [SANDBOX] Filter Wheel connection closed")
        self.session.close()    

        
    def get(self):
        """
        Get the current info from the filter wheel.

        Returns
        -------
        response : str
            Status of the wheel.

        """
        self._purge() # flush
        self.session.write("pos?\r".encode())
        response = self.session.readline().decode()
        
        return response

        
    def get_pos(self):
        """
        Returns the current position of the filter wheel.

        Returns
        -------
        slot : int
            Current position number of the wheel.

        """
        time.sleep(0.1)
        resp = self.get()
        
        if SANDBOX_MODE:
            # Simulated response format: "0 1 COMPLETED"
            parts = resp.split()
            if len(parts) >= 2:
                slot = int(parts[1])  # Take the second part
            else:
                slot = 1  # Default value
        else:
            # Real response format
            slot = int(resp[5])
        
        return slot

    def move(self, slot:int):
        """
        Move the filter wheel to the specified position.

        Parameters
        ----------
        slot : int
            Position number of the wheel to reach.
        """
        if SANDBOX_MODE:
            print(f'⛱️ [SANDBOX] FILT - Move to position {slot}')
        else:
            print('FILT - Move to position '+str(slot))
        self.session.write(("pos="+str(slot)+"\r").encode())
        self.wait()

    
    def wait(self) -> None:
        """
        Wait for the motor to reach the target position.
        """
        if SANDBOX_MODE:
            print("⛱️ [SANDBOX] Filter Wheel - Waiting for position...")
            time.sleep(0.1)  # Reduced wait simulation
            print("⛱️ [SANDBOX] Filter Wheel - Position reached")
        else:
            position = ''
            while len(position) == 0:
                position = self.get()
                time.sleep(0.1)

    
    def wait(self) -> None:
        """
        Wait for the motor to reach the target position.
        """
        position = ''
        while len(position) == 0:
            position = self.get()
            time.sleep(0.1)
