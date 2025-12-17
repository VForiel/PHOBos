from ..photonic_chip import Arch

class Arch15(Arch):
    def __init__(self):
        super().__init__(
            name="Mega Kernel Nuller Reconfig",
            id="N2x2-D4",
            n_inputs=4,
            n_outputs=7,
            topas=(6,7,33,34,35,36,37,38,28,27,26,25,39,40),
            number=15
        )
