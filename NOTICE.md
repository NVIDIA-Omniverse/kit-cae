<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-NvidiaProprietary
-->

# Notices and Attributions

OVCAE  
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

OVCAE is distributed under NVIDIA proprietary terms, matching Kit-CAE. See
[LICENSE.md](LICENSE.md), [PRODUCT_TERMS_OMNIVERSE](PRODUCT_TERMS_OMNIVERSE),
and the license links in [README.md](README.md). This product includes
material from third parties and depends on software distributed under other terms. The following notices
apply as required by their respective licenses.

---

## Bundled Components

### Colormap textures derived from Matplotlib

The textures under `src/ovcae/materials/resources/cae/colormaps/` --
`afmhot.png`, `cividis.png`, `gist_gray.png`, and `gist_rainbow.png` -- were
rasterized from the Matplotlib colormap tables of the same names.

**License:** Matplotlib License (PSF-based)  
**Copyright:** Copyright (c) 2012- Matplotlib Development Team; All Rights Reserved  
**Source:** https://github.com/matplotlib/matplotlib

```
License agreement for matplotlib versions 1.3.0 and later
=========================================================

1. This LICENSE AGREEMENT is between the Matplotlib Development Team
("MDT"), and the Individual or Organization ("Licensee") accessing and
otherwise using matplotlib software in source or binary form and its
associated documentation.

2. Subject to the terms and conditions of this License Agreement, MDT
hereby grants Licensee a nonexclusive, royalty-free, world-wide license
to reproduce, analyze, test, perform and/or display publicly, prepare
derivative works, distribute, and otherwise use matplotlib
alone or in any derivative version, provided, however, that MDT's
License Agreement and MDT's notice of copyright, i.e., "Copyright (c)
2012- Matplotlib Development Team; All Rights Reserved" are retained in
matplotlib  alone or in any derivative version prepared by
Licensee.

3. In the event Licensee prepares a derivative work that is based on or
incorporates matplotlib or any part thereof, and wants to
make the derivative work available to others as provided herein, then
Licensee hereby agrees to include in any such work a brief summary of
the changes made to matplotlib .

4. MDT is making matplotlib available to Licensee on an "AS
IS" basis.  MDT MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR
IMPLIED.  BY WAY OF EXAMPLE, BUT NOT LIMITATION, MDT MAKES NO AND
DISCLAIMS ANY REPRESENTATION OR WARRANTY OF MERCHANTABILITY OR FITNESS
FOR ANY PARTICULAR PURPOSE OR THAT THE USE OF MATPLOTLIB
WILL NOT INFRINGE ANY THIRD PARTY RIGHTS.

5. MDT SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF MATPLOTLIB
FOR ANY INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES OR
LOSS AS A RESULT OF MODIFYING, DISTRIBUTING, OR OTHERWISE USING
MATPLOTLIB , OR ANY DERIVATIVE THEREOF, EVEN IF ADVISED OF
THE POSSIBILITY THEREOF.

6. This License Agreement will automatically terminate upon a material
breach of its terms and conditions.

7. Nothing in this License Agreement shall be deemed to create any
relationship of agency, partnership, or joint venture between MDT and
Licensee.  This License Agreement does not grant permission to use MDT
trademarks or trade name in a trademark sense to endorse or promote
products or services of Licensee, or any third party.

8. By copying, installing or otherwise using matplotlib,
Licensee agrees to be bound by the terms and conditions of this License
Agreement.
```

The `gist_*` colormaps in Matplotlib originate from the Yorick scientific
visualization package (BSD-style license, Copyright (c) 1996 The Regents of
the University of California), and `cividis` was contributed to Matplotlib
by Nuñez, Anderton, and Renslow under the same Matplotlib terms.

### CGNS Static Mixer test data

`tests/data/StaticMixer.cgns` is copied from Warp SimData's test data. It is a
derived form of the Static Mixer example dataset published by the CGNS
project, with mixed element sections expanded into homogeneous sections.

**Dedication:** [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/)

**Contributor:** Richard Hann

**Source:** https://cgns.org/current/examples.html#static-mixer

### Exodus disk-out tutorial data

`tutorials/data/disk_out_ref.ex2` is the ParaView tutorial dataset, produced
at Sandia National Laboratories and redistributed unchanged from the VTK
and ParaView testing data. Its source hash and size are recorded in
[`tutorials/data/README.md`](tutorials/data/README.md).

**License:** BSD 3-Clause License

**Source notice:** https://github.com/Kitware/VTK/blob/master/Copyright.txt

```text
Copyright (c) 1993-2015 Ken Martin, Will Schroeder, Bill Lorensen
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither name of Ken Martin, Will Schroeder, or Bill Lorensen nor the names
   of any contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

---

## Runtime Dependencies

These packages are required to install and run OVCAE. They are not bundled in
the OVCAE wheel; each is installed under its own license terms.

### Warp SimData

**License:** Apache License 2.0  
**Copyright:** Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  
**Source:** https://github.com/NVIDIA/warp-simdata

### warp-lang

**License:** Apache License 2.0  
**Copyright:** Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  
**Source:** https://github.com/NVIDIA/warp

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0

### OVStage

**License:** NVIDIA proprietary (`LicenseRef-NvidiaProprietary`, as declared by
the `ovstage` distribution)  
**Copyright:** Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  
**Source:** https://github.com/NVIDIA-Omniverse/ovstage

OVStage is distributed under NVIDIA's own terms, which accompany that
distribution and are not modified by this notice.

### NumPy

**License:** BSD 3-Clause License  
**Copyright:** Copyright (c) 2005-2024, NumPy Developers. All rights reserved.  
**Source:** https://github.com/numpy/numpy

```
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the NumPy Developers nor the names of any contributors
  may be used to endorse or promote products derived from this software
  without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

---

## Optional Dependencies

### usd-core

**License:** Tomorrow Open Source Technology License 1.0 (`LicenseRef-TOST-1.0`)  
**Copyright:** Copyright 2016 Pixar  
**Source:** https://github.com/PixarAnimationStudios/OpenUSD

### cae-openusd-plugins (`ovcae[usd]`)

**License:** Apache License 2.0  
**Copyright:** Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  
**Source:** https://github.com/NVIDIA-Omniverse/cae-openusd-plugins

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0

---

## Example Application Dependencies (`examples/cae-viewer`)

The CAE Viewer example has its own locked environment. OVCAE does not depend
on these packages.

| Package | License | Source |
| --- | --- | --- |
| OVRTX | NVIDIA proprietary (`LicenseRef-NvidiaProprietary`) | https://github.com/NVIDIA-Omniverse/ovrtx |
| OVUI | NVIDIA proprietary | https://github.com/NVIDIA-Omniverse/ovui |

---

## Test and Development Dependencies

These packages are used only during development and testing and do not ship
with the package.

| Package | License | Copyright / Source |
| --- | --- | --- |
| pytest | MIT | Copyright (c) 2004 Holger Krekel and others; https://github.com/pytest-dev/pytest |
| Ruff | MIT | Copyright (c) 2022 Astral Software Inc.; https://github.com/astral-sh/ruff |
| build | MIT | Copyright (c) 2019 Filipe Laíns; https://github.com/pypa/build |
| VTK | BSD 3-Clause | Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen; https://gitlab.kitware.com/vtk/vtk |
