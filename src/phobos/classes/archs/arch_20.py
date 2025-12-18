from ..photonic_chip import Arch

class Arch20(Arch):
    def __init__(self):
        super().__init__(
            name="2x2 MMI",
            id="MMI2x2-T1",
            n_inputs=2,
            n_outputs=2,
            topas=(),
            number=20
        )
