from ..photonic_chip import Arch

class Arch12(Arch):
    def __init__(self):
        super().__init__(
            name="4x4 MMI Passive",
            id="N4x4-T5",
            n_inputs=4,
            n_outputs=4,
            topas=(),
            number=12
        )
