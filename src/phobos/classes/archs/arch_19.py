from ..photonic_chip import Arch

class Arch19(Arch):
    def __init__(self):
        super().__init__(
            name="2x2 MMI",
            id="N4x4-T2",
            n_inputs=4,
            n_outputs=4,
            topas=(),
            number=19
        )
