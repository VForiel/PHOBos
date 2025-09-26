"""
Mock Serial module for sandbox mode.
Simulates serial communications for equipment like filter wheels,
Zaber and Newport motors.
"""

import time


class Serial:
    """
    Mock of serial.Serial class for sandbox mode.
    Simulates serial communications with equipment.
    """
    
    def __init__(self, port: str, baudrate: int = 9600, timeout: float = 1.0):
        """Simulates opening a serial connection."""
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.is_open = True
        self._buffer = []
        
        print(f"⛱️ [SANDBOX] Opening serial connection: port={port}, baudrate={baudrate}, timeout={timeout}s")
    
    def write(self, data: bytes) -> int:
        """Simulates writing serial data."""
        command = data.decode().strip()
        print(f"⛱️ [SANDBOX] Serial TX ({self.port}): {repr(command)}")
        
        # Prepare a simulated response based on the command
        response = self._generate_response(command)
        if response:
            self._buffer.append(response.encode() + b'\r\n')
        
        return len(data)
    
    def readline(self) -> bytes:
        """Simulates reading a line from the serial port."""
        if self._buffer:
            response = self._buffer.pop(0)
            response_str = response.decode().strip()
            print(f"⛱️ [SANDBOX] Serial RX ({self.port}): {repr(response_str)}")
            return response
        else:
            # Return empty response if nothing in buffer
            print(f"⛱️ [SANDBOX] Serial RX ({self.port}): (empty)")
            return b''
    
    def readlines(self) -> list:
        """Simulates reading multiple lines."""
        lines = []
        while self._buffer:
            lines.append(self.readline())
        return lines
    
    def close(self):
        """Simulates closing the serial connection."""
        if self.is_open:
            print(f"⛱️ [SANDBOX] Closing serial connection: {self.port}")
            self.is_open = False
    
    def _generate_response(self, command: str) -> str:
        """Generates a simulated response based on received command."""
        command = command.upper().strip()
        
        # Responses for Filter Wheel (Thorlabs)
        if command == "POS?":
            return "0 1 COMPLETED"  # Position 1
        elif command.startswith("POS="):
            pos = command.split('=')[1]
            return f"0 {pos} COMPLETED"
        
        # Responses for Zaber motors
        elif command.startswith("/"):
            parts = command.split()
            if len(parts) >= 2:
                axis_str = parts[0][1:]  # Remove the '/'
                try:
                    axis = int(axis_str)
                except ValueError:
                    axis = 1
                cmd = parts[1].upper()
                
                if cmd == "GET" and len(parts) >= 3 and parts[2].upper() == "POS":
                    return f"@{axis:02d} 0 OK IDLE -- 150000"  # Simulated position
                elif cmd == "MOVE":
                    if len(parts) >= 4:
                        move_type = parts[2].upper()  # ABS or REL
                        position = parts[3]
                        return f"@{axis:02d} 0 OK BUSY -- {position}"
        
        # Responses for Newport
        elif command == "1TP?":
            return "1TP56.150"  # Simulated angular position
        elif command == "1OR?":
            return "1OR"  # Home search
        elif command.startswith("1PA"):
            pos = command[3:]
            return f"1PA{pos}"
        elif command.startswith("1PR"):
            pos = command[3:]
            return f"1PR{pos}"
        
        # Generic response
        return "OK"