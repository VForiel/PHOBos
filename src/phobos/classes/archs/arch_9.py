from ..photonic_chip import Arch

class Arch9(Arch):
    def __init__(self):
        super().__init__(
            name="Normal 4-Port Nuller (active 2x2)",
            id="N2x2-T6",
            n_inputs=4,
            n_outputs=4,
            topas=(21,22,23,24,25,26,27,28),
            number=9
        )
