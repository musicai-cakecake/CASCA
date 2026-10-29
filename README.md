<p align="center">
<img src="assets/qq_studio.png" alt="CUC, Peking University, Tencent." width="400" />
</p>

<h2 align="center">CASCA: Learning Cross-Axis Dependency and Sparsity Calibration<br>with Discrete Diffusion for Unified Symbolic Music Generation</h2>

<p align="center"><strong>An end-to-end unified base model for intelligent creation platforms.</strong></p>

---
## Introduction
CASCA is a unified base model for intelligent creation platforms. It generates multi-instrument accompaniment or target instrument tracks conditioned on input tracks and prompts.
Its data representation is compatible with intelligent creation platforms and digital audio workstations (DAWs) for human–AI collaborative creation scenarios.

The main features are as follows:  

- **End-to-end inference**: Performs end-to-end generation of multi-instrument accompaniment. No prior knowledge of input tracks, expert systems, or manual domain expertise is required.
- **Prompt-conditioned multi-instrument accompaniment**: Supports generating multi-instrument accompaniment, where each instrument can be controlled by its dedicated prompt.
- **Editable generation & Human–AI collaborative creation**: Users specify target instruments and score regions to enable editable generation. Human-authored score segments guide generation and remain preserved.
- **Native integration with intelligent creation platforms**: The data representation works with score editors of intelligent creation platforms and digital audio workstations (DAWs), with no need to encode local MIDI files.
- **Dataset-aligned calibration of music event distribution**: The ratio of PAD tokens to musical events is calibrated against the training dataset.

<p align="center">
<img src="assets/editable.png" alt="Editable generation" width="600" />
</p>

As shown above, CASCA supports editable generation and serves as a base model for intelligent creation platforms. For instance, users may prepare a lead melody (visualized as dark-blue segments).
If users are unable to compose bass and drum parts, CASCA automatically generates these tracks on demand.
Generated data can be delivered via APIs directly to editors of DAWs or agents. Local digital score files or disk-stored intermediate artifacts (e.g., MIDI, XML, ABC) are not required.

Leveraging these capabilities, CASCA can be deployed for assisted music creation and art education use cases. The schematic diagram is shown below:

<p align="center">
<img src="assets/application_diagram.png" alt="Schematic diagram of applications" width="420" />
</p>

Main contributions and innovations:

- A unified perspective on symbolic music generation highlights two complementary requirements: cross-axis dependency learning for cross-track coordination and temporal continuity, and sparsity calibration to counter the over-generation of empty-time states observed in discrete diffusion under extreme data imbalance.

- A block discrete diffusion framework, CASCA, integrates a unified 2D representation, row-adaptive attention, factorized 2D rotary position embeddings, and region-aware masking for multi-instrument generation and editing.

- A sparsity-calibrated loss, PFSUBS, with confidence-aware gradient calibration, controls the proportions of empty-time and music-event tokens toward a prescribed target distribution. It reverses the generation bias toward empty time and increases musical richness while preserving full-vocabulary reconstruction.

- A large-scale multi-instrument corpus, together with parallel training and cached inference, supports large-vocabulary modeling. A comprehensive experimental toolkit is constructed for evaluating generation quality, conditional preservation, and sparsity calibration.

---
## Demo
Several generated samples along with their corresponding ground truth (GT) are available in the Demos folder.  
We do not provide audio rendering, as our model only outputs symbolic scores. 
This work focuses on human–AI interactive composition of score content; audio quality is outside the scope of our research. The final quality of released music heavily depends on timbre, mixing, and post-production. Audio exported using default DAW sound libraries tends to have poor musical expressiveness, which can be mistakenly attributed to defects in the generated score.

An example generation is shown below. The first track is the input melody named lead, while the remaining tracks constitute the generated accompaniment. For this sample, all six supported instrument types of our model are specified for generation.
The input prompt is: Allegro angry jazz lead piano guitar bass string wind drum.
<p align="center"> <img src="Demos/592/592_gen.png" alt="Schematic diagram of applications" width="1200" /> </p>
The corresponding GT of this sample is provided below:
<p align="center"> <img src="Demos/592/592_GT.png" alt="Schematic diagram of applications" width="1200" /> </p>

The source file of this sample is located at: Demos/592

---
## Resource
The two versions mentioned in the paper are both available in this repository.
CASCA is trained from random initialization with the original SUBS objective.
CASCA-PFSUBS receives 225 post-training updates with the proposed Padding-Aware Focal Substitution-Based (PFSUBS) loss.

The model ckpt for CASCA can be obtained from the following link:   [google drive](https://drive.google.com/file/d/1ScDReYZc6MeCbLMJ0pSaWb91qvZP9UvQ/view?usp=sharing).

The processed representation can be downloaded via [google drive](https://drive.google.com/file/d/1-HKIbJX8T_XbMBMdtkfjx8JVcisfBTO6/view?usp=sharing).  
It is processed on the XMIDI dataset. The original XMIDI is accessible via [google drive](https://drive.google.com/file/d/1qDkSH31x7jN8X-2RyzB9wuxGji4QxYyA/view?usp=sharing).
