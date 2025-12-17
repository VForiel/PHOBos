from ..photonic_chip import Arch

class Arch13(Arch):
    def __init__(self):
        super().__init__(
            name="1x2 MMI Passive",
            id="MMI1x2-T4",
            n_inputs=1,
            n_outputs=2,
            topas=(),
            number=13
        )
