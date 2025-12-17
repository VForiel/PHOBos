from ..photonic_chip import Arch

class Arch14(Arch):
    def __init__(self):
        super().__init__(
            name="Phase Actuator Solo",
            id="PM-T3",
            n_inputs=1,
            n_outputs=1,
            topas=(16,),
            number=14
        )
