
try:
    import kbench
    print(f"kbench imported: {kbench}")
    print(f"kbench.classes: {kbench.classes}")
    
    from kbench.classes import DM
    print(f"DM imported from kbench.classes: {DM}")
    
    from kbench.classes.deformable_mirror import DM as DM2
    print(f"DM imported from kbench.classes.deformable_mirror: {DM2}")

except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
