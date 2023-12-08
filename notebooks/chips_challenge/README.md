# Getting Started Tutorial

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](https://openvinotoolkit.github.io/anomalib/getting_started/installation/index.html).
You will also need OpenVINO, so make sure you install that option.

## Notebook Contents

This notebook demonstrates the usage of anomaly detection in a real-world use case of anomaly detection on pharmaceutical objects.
We use the Sensum Solid Oral Dosage Forms (Sensum SODF) dataset that contains two forms of solid oral dosage forms.
We use only the first form - a capsule.

![image](https://github.com/openvinotoolkit/anomalib/assets/61357777/8dc9abee-7c4f-4994-b35b-7c7b7dac5b41)

Notebook contains the following sections:

- Installing Anomalib
- Configuration
- Dataset Preparation
- Model Preparation
- Model Training and Validation
- OpenVINO Inference

---

SensumSODF is shared under CC BY-NC-SA 4.0 licence. Request for download at: https://www.sensum.eu/sensumsodf-dataset/. 

Dataset citation:
```
@article{Racki2021NCAA,
    author = {Ra{\v{c}}ki, Domen and Toma{\v{z}}evi{\v{c}}, Dejan and Sko{\v{c}}aj, Danijel},
    title = {Detection of surface defects on pharmaceutical solid oral dosage forms with convolutional neural networks},
    journal = {Neural Computing and Applications},
    year = {2021},
    month = {August},
    day = {17},
    issn = {1433-3058},
    doi = {10.1007/s00521-021-06397-6}
}
```

