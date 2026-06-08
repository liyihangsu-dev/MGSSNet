# PSSR

**Progressive Multi-Granularity Invertible Semantic-Spatial Fusion for Multi-Contrast MRI Super-Resolution**
--------------------------------------------------------------------------------------------------------------------

## Abstract

Magnetic Resonance Imaging (MRI) has become a core clinical imaging modality for disease diagnosis and treatment planning owing to its non-invasive and non-ionizing properties. Multi-contrast MRI sequences inherently provide complementary anatomical and pathological information for comprehensive clinical diagnosis, but long scanning time and physiological motion artifacts often lead to low-resolution target modal images, posing a reconstruction challenge. Single-contrast super-resolution methods, relying solely on the target modality, fail to recover fine anatomical details (e.g., ventricles and sulci). Existing multi-contrast methods suffer from insufficient high-low frequency feature decoupling, poor adaptability to cross-modal gray-scale heterogeneity, and ineffective dual-branch collaboration, limiting their performance and clinical utility.To solve these issues, this paper proposes a pixel-semantic dual-branch collaborative network (PSSR) for multi-contrast MRI super-resolution, enabling high-precision restoration via semantic-pixel synergy. Its parallel dual-branch architecture includes: a semantic branch that explicitly decouples high-low frequency features via discrete wavelet transform, suppresses noise, and corrects cross-modal spatial offset; a pixel branch that models cross-modal correlations via dual LoRA low-rank branches to generate adaptive convolution kernels for detail reconstruction; and a gradient reassignment strategy to alleviate dual-branch gradient decay and enhance collaboration. Experiments on multiple public MRI datasets show that PSSR outperforms state-of-the-art methods in quantitative metrics across various upsampling ratios with excellent generalization. It effectively restores anatomical details and textures of low-resolution MRI, offering a new technical solution for clinical multi-contrast MRI super-resolution.Multi-contrast MRI sequences provide complementary anatomical and pathological information for clinical diagnosis. However, long scanning time and physiological motion often lead to low-resolution target modal images. This paper proposes PSSR, a pixel-semantic dual-branch collaborative network for multi-contrast MRI super-resolution. The semantic branch explicitly decouples high-low frequency features via discrete wavelet transform (DWT), suppresses noise through semantic mixture (SE attention), and corrects cross-modal spatial offset via semantic alignment. The pixel branch adopts dual LoRA-driven dynamic convolution modules (DSPM) to model cross-modal correlations and generate adaptive kernels for detail reconstruction. A gradient reassignment strategy (GRS) routes auxiliary loss gradients directly to semantic blocks, alleviating gradient decay in deep dual-branch training.

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