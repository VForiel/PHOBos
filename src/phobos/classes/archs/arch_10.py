from ..photonic_chip import Arch

class Arch10(Arch):
    def __init__(self):
        super().__init__(
            name="4-Port Nuller (4x4 MMI) Passive Crazy",
            id="N4x4-D6",
            n_inputs=4,
            n_outputs=7,
            topas=(),
            number=10
        )
