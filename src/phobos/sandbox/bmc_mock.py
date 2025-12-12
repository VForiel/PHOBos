"""
Mock BMC module for sandbox mode.
Simulates BMC library functionalities for testing and development.
"""

# Simulated BMC constants
DM_Piston = 0
DM_XTilt = 1
DM_YTilt = 2

class BmcDm:
    """
    Mock of BmcDm class for sandbox mode.
    Displays commands that would be sent to the deformable mirror.
    """
    
    def __init__(self):
        self.is_open = False
        self.serial_number = None
        
    def open_dm(self, serial_number: str):
        """Simulates opening a DM connection."""
        self.serial_number = serial_number
        self.is_open = True
        print(f"⛱️ [SANDBOX] Opening DM connection: {serial_number}")
        return True
        
    def close_dm(self):
        """Simulates closing a DM connection."""
        if self.is_open:
            print(f"⛱️ [SANDBOX] Closing DM connection: {self.serial_number}")
            self.is_open = False
            self.serial_number = None
        
    def set_segment(self, segment_id: int, piston: float, tip: float, tilt: float, 
                   absolute: bool = True, apply: bool = True) -> str:
        """Simulates segment configuration."""
        mode = "absolute" if absolute else "relative"
        apply_str = "applied" if apply else "staged"
        
        print(f"⛱️ [SANDBOX] DM {self.serial_number} - Set segment {segment_id}: "
              f"piston={piston:.3f}nm, tip={tip:.6f}rad, tilt={tilt:.6f}rad "
              f"({mode}, {apply_str})")
        
        return f"Segment {segment_id} configured successfully"
        
    def get_segment_range(self, segment_id: int, parameter_type: int, 
                         piston: float, tip: float, tilt: float, absolute: bool = True) -> list[float]:
        """Simulates retrieval of segment parameter limits."""
        param_names = {DM_Piston: "piston", DM_XTilt: "tip", DM_YTilt: "tilt"}
        param_name = param_names.get(parameter_type, f"param_{parameter_type}")
        
        # Simulated limit values
        ranges = {
            DM_Piston: [-2000.0, 2000.0],  # nm
            DM_XTilt: [-0.01, 0.01],       # rad
            DM_YTilt: [-0.01, 0.01]        # rad
        }
        
        range_values = ranges.get(parameter_type, [-1000.0, 1000.0])
        
        print(f"⛱️ [SANDBOX] DM {self.serial_number} - Get range for segment {segment_id} "
              f"{param_name}: {range_values}")
        
        return range_values