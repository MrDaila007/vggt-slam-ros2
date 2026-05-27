# Third-Party Licenses

This package integrates with the following third-party projects.
Their licenses are reproduced below for reference.

---

## 1. VGGT — Visual Geometry Grounded Transformer

**Repository:** https://github.com/facebookresearch/vggt  
**Authors:** Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, David Novotny (VGG Oxford / Meta AI)  
**Paper:** VGGT: Visual Geometry Grounded Transformer, CVPR 2025 (Best Paper Award)

### How we use it

We import VGGT as a Python dependency (`from vggt.models.vggt import VGGT`, etc.).
No VGGT source code or model weights are bundled in this repository.

### ⚠️ License restrictions you must know

VGGT source code **and** model weights are distributed under Meta's custom
**VGGT Research License** (not Apache-2.0, not MIT).

| Checkpoint | Commercial use | Notes |
|---|---|---|
| `facebook/VGGT-1B` (default) | **NO** — research/non-commercial only | Freely downloadable from HuggingFace |
| `facebook/VGGT-1B-Commercial` | **YES** — with Meta approval | Requires an application via HuggingFace |

**Military and weapons applications are explicitly prohibited** for both checkpoints.

If you use this package for commercial purposes you must:
1. Submit an application for `VGGT-1B-Commercial` at HuggingFace
2. Set `checkpoint: "facebook/VGGT-1B-Commercial"` in `config/params.yaml`

The full VGGT license text is at:
https://github.com/facebookresearch/vggt/blob/main/LICENSE.txt

---

## 2. VGGT-SLAM

**Repository:** https://github.com/MIT-SPARK/VGGT-SLAM  
**Authors:** Dominic Maggio, Luca Carlone (MIT SPARK Lab)  
**License:** BSD 2-Clause

### How we use it

We do **not** use any VGGT-SLAM code.
VGGT-SLAM served as architectural inspiration; the sliding-window pipeline,
ROS2 integration, and all code in this repository were written independently.

### BSD 2-Clause License

```
Copyright (c) 2025, MIT-SPARK

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
