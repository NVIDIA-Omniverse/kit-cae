# Tutorial data

Scenes and datasets used by the tutorials. Run `git lfs pull` after cloning
to fetch the CGNS and Exodus files. Tutorial A1 uses the
[Static Mixer test dataset](../../tests/data/README.md) instead.

## auto_aero_solver_result.cgns, aero_streamlines.usda, aero_isosurface.usda, aero_glass.mdl

The main-guide dataset of the [Kit-CAE user guide][kit-cae-guide]: a
steady-state external-aerodynamics simulation around a concept vehicle, as a
CGNS file (341,609,342 bytes, Git LFS) with cell-associated pressure and
velocity on about 7.9 million nodes and 7.2 million polyhedral cells, plus
the body, wheel, and wheel-seal surface zones. Two recipe scenes payload it:
`aero_streamlines.usda` (tutorial 1: body, wheel surfaces, streamlines) and
`aero_isosurface.usda` (tutorial 2: pressure iso-surface bound to the custom
glass material in `aero_glass.mdl`). Tutorial 3 reads the CGNS file alone.
Keep the four files side by side; the scenes reference the file and the
material by relative path.

- Source: NVIDIA. The file is `main_guide/auto_aero_solver_result.cgns` in
  the [Kit-CAE user guide data archive][kit-cae-data] and is redistributed
  here unchanged. The scenes were authored in Kit-CAE and extended with a
  camera and render product for these tutorials.
- License: distributed with the Kit-CAE user guide under the
  [NVIDIA Omniverse License Agreement][omniverse-license].

[kit-cae-guide]: https://docs.omniverse.nvidia.com/guide-kit-cae/latest/v3/getting-started.html#main-guide-dataset
[kit-cae-data]: https://d4i3qtqj3r0z5.cloudfront.net/Kit_CAE_user_guide_data%4010010.zip
[omniverse-license]: https://docs.omniverse.nvidia.com/install-guide/latest/common/NVIDIA_Omniverse_License_Agreement.html

## disk_out_ref.ex2

The ParaView tutorial dataset: a steady-state simulation of gas flow around
a heated, rotating disk, stored as an Exodus II file. One hexahedral block
of 8,499 nodes and 7,472 elements with the nodal fields `Temp`, `Pres`, the
velocity vector `V`, and the species fractions `AsH3`, `GaMe3`, `CH4`, and
`H2`.

- Source: the VTK and ParaView testing data published by Kitware, fetched
  from `https://www.vtk.org/files/ExternalData/SHA512/<hash>`.
- SHA-512: `d66c52c6b7ad3766ac2693bab2cc773ac43d18d753742764498ea7fab4c3e3cb06c58d0d922977c46a6ed7c59868700b7a2faba9f73c40b71caff62d1f9eb0a9`
- Size: 704,400 bytes, stored through Git LFS.
- License: BSD 3-Clause. The full copyright notice, conditions, and
  disclaimer from VTK's `Copyright.txt` are included in
  [NOTICE.md](../../NOTICE.md#exodus-disk-out-tutorial-data). The simulation
  was produced at Sandia National Laboratories and is redistributed here
  unchanged. Retain the full notice when redistributing the dataset.

Appendix A2 reads it with VTK's Exodus II reader through a custom Data
Source.
