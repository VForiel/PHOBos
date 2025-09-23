# Configuration File

The configuration file is a YAML file that contains the settings for the Kbench software. It is used to define the hardware connections and the position of the different elements.

File example:

```yaml
# Mask wheel ------------------------------------------------------------------

mask:
  ports:
    newport: '/dev/ttyUSB1' # Newport wheel USB port
    zaber: '/dev/ttyUSB2'   # Zabers USB port

  slots:                    # Virtual slots 
    'dot':                  # Name of the mask
      x: 0                  # X-position in ticks
      y: 0                  # Y-position in ticks
      a: 0.0                # Angle in degree relatively
      
# Filter wheel ----------------------------------------------------------------

filter:
  port: '/dev/ttyUSB0'      # Filter wheel USB port

  slots:                    # Virtual slots
    'Density50%':           # Name of the filter
        base_slot: 1        # Physical slot
```

Virtual slots allows to abstract the physical slots of the wheels and create as many slots as you want according to the scenario. For example, you can create a virtual slot "dot" that corresponds to the physical slot 1 of the mask wheel, and a slot "shifted_dot" that is also in the physical slot 1 but with a different position of the zabers.