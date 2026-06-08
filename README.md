# PSSR

**Progressive Multi-Granularity Invertible Semantic-Spatial Fusion for Multi-Contrast MRI Super-Resolution**
--------------------------------------------------------------------------------------------------------------------

## Abstract

Multi-Contrast Magnetic Resonance Imaging Super-Resolution leverages structural priors from a high-resolution reference modality to guide target image reconstruction. However, existing paradigms suffer from three fundamental bottlenecks: 1) frequency-domain feature coupling that causes aliasing of noise and contours; 2) non-linear spatial misalignment and intensity heterogeneity that hinder domain gap elimination; and 3) severe gradient attenuation in dual-branch topologies that decouples feature alignment from pixel reconstruction. To address these challenges, we propose the Multi-Granularity Invertible Semantic-Spatial Super-Resolution Network (MGSSNet), a progressive alignment architecture featuring frequency awareness, long-range spatial mapping, and lossless gradient backpropagation. Specifically, the Spatial Cross-Granularity Alignment Module (SCGAM) integrates multi-scale convolutions with the Discrete Wavelet Transform (DWT) to orthogonally disentangle cross-modality features into frequency sub-bands, injecting semantic guidance into low-frequency components to achieve explicit spatial alignment while suppressing noise propagation. Concurrently, the Invertible Semantic-Spatial Alignment Module (ISSAM) constructs a cascaded spatial-semantic hybrid Mamba topology, which deploys parameter-shared 2D State Space (SS2D) operators to correct non-linear misalignments with linear complexity, while exploiting bidirectional invertible coupling to guarantee degradation-free gradient flow. Extensive experiments on BraTS, IXI, and CHAOS datasets demonstrate that MGSSNet establishes a new state-of-the-art performance ceiling and exhibits superior clinical structure preservation in zero-shot cross-domain scenarios.

---

## Quick Start

### 1. Training

To train the model on the specified dataset from scratch, execute the following command:

```bash
python main.py \
  --model MGSSNet \
  --dataset BraTSReg \
  --upscale 4 \
  --batch_size 8 \
  --image_size 256

```

### 2. Evaluation / Testing

To evaluate the trained model checkpoint, use the verification script:

```bash
python val.py \
  --model MGSSNet \
  --dataset BraTSReg \
  --upscale 4 \
  --checkpoint_path ./checkpoints/best_model.pth

```

---

### Datasets

| Dataset | Modality | Description | Official Link |
| --- | --- | --- | --- |
| **BraTSReg** | Brain MRI (T1 / T2) | Primary benchmark for multi-contrast MRSR | [🔗 Website](https://bratsreg.github.io/) |
| **IXI** | Brain MRI (T1 / T2) | Generalization and structural consistency test | [🔗 Website](http://brain-development.org/ixi-dataset/) |
| **CHAOS** | CT / MRI | Robustness evaluation via **zero-shot** testing | [🔗 GitHub](https://github.com/Monk007codes/Pgm-CHAOS) |
