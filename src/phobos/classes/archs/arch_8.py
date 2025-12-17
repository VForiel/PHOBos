from ..photonic_chip import Arch

class Arch8(Arch):
    def __init__(self):
        super().__init__(
            name="Mach-Zender Interferometer",
            id="MZI-T7",
            n_inputs=1,
            n_outputs=1,
            topas=(4,5),
            number=8
        )
