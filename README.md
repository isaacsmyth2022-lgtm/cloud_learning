# cloud_learning
This repository contains the publishable work from my first year dissertation. As this method of cloud generation may be useful for others in the field, I felt it best to open-source it.

The Cloud Generator file must be opened using Blender. I used Blender version 5.0.1, and results may differ by version as Blender are currently updating their OpenVDB capabilities. 

**cloud_generator.py** — Synthetic cloud shadow dataset generator built on top of Blender/Cycles.

Procedurally generates volumetric cloud fields as OpenVDB files and bakes their ground shadows to images using Blender's renderer. Cloud properties are physically grounded: liquid water content, droplet radius, and geometric thickness are sampled from realistic ranges for eight WMO cloud types (Cu fra, Cu hum, Cu med, Cu con, Cb, Sc, St, Ns), with optical depth computed from standard cloud microphysics before calibrating the voxel densities that Cycles renders.

Three shape generators handle different cloud morphologies: puffy convective cumulus, flat stratiform sheets, and small scattered cloudlets. Each uses multi-lobe Gaussian envelopes, domain warping, and layered noise to produce plausible structure without running a full fluid sim. There's also an optional wind mode that keyframes clouds drifting across the scene for generating temporal shadow sequences.
Each sample outputs a paired JPEG shadow bake and JSON file recording the microphysical parameters used.


**train_classifier.py** — Trains classifiers to distinguish cloud shadow images across four classes (cumulus/stratocumulus × low/high tau).

Loads greyscale shadow bakes from a folder of class subfolders, splits them into train/test sets, and runs two approaches back to back. The first extracts a set of handcrafted features per image (intensity statistics, spatial contrast, gradient energy, FFT frequency content, radial brightness profiles) and trains a Random Forest and a small MLP on top. The second trains a three-block CNN in PyTorch directly on the raw pixel values, with batch norm, dropout, and a learning rate scheduler. Both print accuracy, per-class precision/recall, and a confusion matrix at the end.

Sklearn classifiers run by default; PyTorch is optional and skips gracefully if not installed. The trained CNN saves to shadow_classifier.pth.

**Bakes**
These JPG files are random examples of stratiform cloud scene shadow bakes. These are the end result of Blender simulation.
