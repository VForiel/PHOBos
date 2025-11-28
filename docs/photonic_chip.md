# Photonic Chip

```{eval-rst}
.. automodule:: kbench.classes.photonic_chip
   :members:
   :show-inheritance:
```

## XPOW Controller

The photonic chip is controlled via the **XPOW-8AX-CCvCV-U** controller from NICSLAB. 
This controller provides 40 independent channels with voltage and current control for thermo-optic phase actuators (TOPAs).

- **Product documentation**: [XPOW Datasheet](https://www.nicslab.com/product-datasheet)
- **Model**: XPOW-8AX-CCvCV-U
- **Interface**: Serial communication (115200 baud)
- **Channels**: 40 independent voltage/current sources
- **Voltage range**: 0-5V
- **Current range**: 0-300mA

## Available Chip Architectures

The `Chip` class supports 20 different photonic chip architectures, each with specific configurations of inputs, outputs, and thermo-optic phase actuators (TOPAs):

### Mach-Zehnder Interferometers (MZI)

| Arch | ID | Name | Inputs | Outputs | TOPAs |
|------|----|----|--------|---------|-------|
| 1 | MZI-T12 | Mach-Zehnder Interferometer | 1 | 1 | 1, 2 |
| 8 | MZI-T7 | Mach-Zender Interferometer | 1 | 1 | 4, 5 |

### Phase Shifters/Modulators

| Arch | ID | Name | Inputs | Outputs | TOPAs |
|------|----|----|--------|---------|-------|
| 2 | PM-T11 | Phase Shifter Solo | 1 | 1 | 3 |
| 14 | PM-T3 | Phase Actuator Solo | 1 | 1 | 16 |

### Multi-Mode Interferometers (MMI)

| Arch | ID | Name | Inputs | Outputs | TOPAs |
|------|----|----|--------|---------|-------|
| 3 | MMI2x2-T10 | 2x2 MMI Solo | 2 | 2 | - |
| 4 | MMI1x2-T9 | 1x2 MMI Solo | 1 | 2 | - |
| 12 | N4x4-T5 | 4x4 MMI Passive | 4 | 4 | - |
| 13 | MMI1x2-T4 | 1x2 MMI Passive | 1 | 2 | - |
| 19 | N4x4-T2 | 2x2 MMI | 4 | 4 | - |
| 20 | MMI2x2-T1 | 2x2 MMI | 2 | 2 | - |

### Kernel Nullers (2x2 configurations)

| Arch | ID | Name | Inputs | Outputs | TOPAs |
|------|----|----|--------|---------|-------|
| 9 | N2x2-T6 | Normal 4-Port Nuller (active 2x2) | 4 | 4 | 21-28 |
| 15 | N2x2-D4 | Mega Kernel Nuller Reconfig | 4 | 7 | 6, 7, 25-40 |
| 16 | N2x2-D3 | Kernel Nuller 2x2 Reconfig N | 4 | 7 | 29-32 |
| 17 | N2x2-D2 | Passive Kernel Nuller | 4 | 7 | - |

### Kernel Nullers (3x3 configurations)

| Arch | ID | Name | Inputs | Outputs | TOPAs |
|------|----|----|--------|---------|-------|
| 11 | N3x3-D5 | 3-Port Kernel Nuller (passive) | 3 | 3 | - |
| 18 | N3x3-D1 | 3-Port Kernel Nuller | 3 | 3 | - |

### Kernel Nullers (4x4 configurations)

| Arch | ID | Name | Inputs | Outputs | TOPAs |
|------|----|----|--------|---------|-------|
| 5 | N4x4-D8 | 4-Port Nuller Reconfig | 4 | 7 | 4-16 |
| 6 | N4x4-T8 | 4-Port MMI Active | 4 | 4 | 17-20 |
| 7 | N4x4-D7 | 4-Port Nuller (passive) - FT | 4 | 7 | 21-24 |
| 10 | N4x4-D6 | 4-Port Nuller (4x4 MMI) Passive Crazy | 4 | 7 | - |

:::{note}
Architecture 6 (N4x4-T8) is commonly used for 4-port kernel nulling interferometry with active phase control on 4 output channels.
:::

:::{tip}
Use `Chip.ARCHS[arch_number]` to access the full architecture dictionary for a specific configuration.
:::