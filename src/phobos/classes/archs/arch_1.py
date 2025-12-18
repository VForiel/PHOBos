from ..photonic_chip import Arch

class Arch1(Arch):
    def __init__(self):
        super().__init__(
            name="Mach-Zehnder Interferometer",
            id="MZI-T12",
            n_inputs=1,
            n_outputs=1,
            topas=(1,2),
            number=1
        )
