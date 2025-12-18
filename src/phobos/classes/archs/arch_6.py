from ..photonic_chip import Arch

class Arch6(Arch):
    def __init__(self):
        super().__init__(
            name="4-Port MMI Active",
            id="N4x4-T8",
            n_inputs=4,
            n_outputs=4,
            topas=(17,18,19,20),
            number=6
        )
