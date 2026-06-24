# LimX Dynamics HUD03

The MJCF and mesh assets in this package are derived from the
`HU_D03_description` robot description supplied with this repository.

`hud03_constants.get_spec()` converts the source closed-chain Achilles and waist
transmissions into a 31-DoF serial training articulation. It removes the auxiliary
ball/driver joints and equality constraints, keeps their bodies rigidly attached so
the original mass and inertia are preserved, and actuates the ankle and waist output
joints directly. This matches MjLab's scalar-joint state assumptions and the reduced
parallel-linkage abstraction used by the Unitree G1 locomotion model.
