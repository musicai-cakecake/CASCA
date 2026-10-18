

<p align="center">
<img src="assets/qq_studio.png" alt="CUC, Peking University, Tencent." width="400" />
</p>

<h2 align="center">CASCA: Learning Cross-Axis Dependency and Sparsity Calibration<br>with Discrete Diffusion for Unified Symbolic Music Generation</h2>

<p align="center"><strong>An end-to-end unified base model for the intelligent creation platform.</strong></p>


CASCA is a unified base model for intelligent creation platforms. It generates multi-instrument accompaniment or target instrument tracks from given input tracks and prompts.
The data representation are compatible with intelligent creation platforms and digital audio workstations (DAWs) for human–AI collaborative creation scenarios.   
The main features are as follows:  

- **End-to-end inference**: End-to-end generation of multi-instrument accompaniment, no need for priori knowledge of the input tracks, nor human knowledge and expert systems.
- **Multi-instrument accompaniment on respective prompts**: Generate multi-instrument accompaniment, each instrument has its own prompt.
- **Editable generation and Human–AI collaborative creation**: Specify the wanted instruments and score boundaries to satisfy the Editable generation. The human input score is used to guide the generation and remains unchanged.
- **Applicable to intelligent creation platforms**: The data representation are compatible with the score editing window of the intelligent creation platform, and the digital audio workstation (DAW), do not encode local MIDI files.
- **The distribution of music events is calibrated with respect to the dataset**: The PAD/event proportions has been calibrated based on the dataset.

<p align="center">
<img src="assets/editable.png" alt="Editable generation." width="600" />
</p>

As shown above, CASCA supports editable generation as a base model for intelligent creation platforms. For instance, users can prepare a lead melody (visualized as dark-blue segments); CASCA automatically generates bass and drum tracks on request if users cannot compose them. Generated data is sent via APIs directly to editors of DAWs or agents, without requiring local digital score files or disk-stored intermediate files such as MIDI, XML, or ABC. 

