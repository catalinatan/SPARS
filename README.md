SPAR-WSSS: Self-Play Adversarial Reinforcement Learning for Weakly-Supervised Semantic Segmentation

### Abstract
Accurate tumour segmentation is vital for targeted diagnostic and
therapeutic surgical procedures for cancer. Current tumour segmentation
methods involve manual delineation which is both labour-intensive and
subjective. Fully-supervised machine learning models aim to address
these issues, but require a large number of costly and often subjective
3D-voxel level labels for training. In this work, we propose a novel
framework called SPAR-WSSS (Self-Play Adversarial Reinforcement
Learning for Weakly-Supervised Semantic Segmentation), which utilises
an object presence classifier, trained on a small number of image-level
binary cancer presence labels, to localise cancerous regions on CT scans.
Such binary labels of patient-level cancer presence can be sourced more
objectively from biopsies and histopathology reports, enabling a more
objective cancer localisation on medical images. Evaluating with real
patient data, we observed that SPAR-WSSS yielded a mean dice score of
77.3 ± 9.4, which outperformed other weakly-supervised methods by
large margins. This performance was comparable with recent fully-
supervised methods that require voxel-level annotations. Our results
demonstrate the potential of using SPAR-WSSS to reduce the need for
extensive human-annotated labels to detect cancer in real-world
healthcare settings.

### Method
We propose an adversarial reinforcement learning environment for
weakly-supervised segmentation where two agents compete to identify
ROIs in images. During training, each agent moves a window across the
image at every time step and receives a score from an object presence
classifier, which is pre-trained using binary, image-level labels of object
presence. This score determines the likelihood that the ROI falls within
this area and drives the reward signal to train the agents to maximise
localisation accuracy. An agent will receive a positive reward if its score
exceeds that of the other agent, and receive a negative reward otherwise.
The probability scores of each voxel in the segmentation map is initially
set to zero. As a window moves across a region in an image, the voxel
values within the window are updated to reflect the score received by the
classifier. These scores continue to accumulate until one of the agents
receives a predetermined score at a single time step or when a predefined
number of iterations has been completed. When the algorithm reaches
this termination condition, the agent assumes that a sufficient
localisation performance has been reached.

![Method Overview](Method.png)

### Algorithmn Parameters
# Object Presence Classifier Parameters

| **Parameters**            | **Details**           |
|---------------------------|-----------------------|
| Convolution layers         | 4                     |
| Fully connected layers     | 5                     |
| Input-size                 | 256 x 256 x 180       |
| Batch-size                 | 4                     |
| Epochs                     | 32                    |
| Optimiser                  | Adam                  |
| Criterion                  | Cross Entropy Loss    |
| Learning rate              | 0.001                 |

# SPAR-WSSS Parameters

| **Parameters**               | **Details**                   |
|------------------------------|-------------------------------|
| Input-size*                   | 256 x 256 x 180              |
| Batch-size                    | 8                            |
| Epoch                         | 1                            |
| Agent Model                   | Proximal Policy Optimisation |
| Competitor Update Frequency** | 32                           |
| Iterations                    | 10,000                       |
| Action                        | 4-voxel span                 |

*Input size denotes the dimensions of the data used in the model.  
**Competitor Update Frequency refers to how often the competitor model is updated during training.

### Setting Up a Virtual Environment

1. **Create a virtual environment**:
    ```sh
    python3 -m venv venv
    ```

2. **Activate the virtual environment**:
    - On macOS and Linux:
        ```sh
        source venv/bin/activate
        ```
    - On Windows:
        ```sh
        venv\Scripts\activate
        ```

## Installing the Project
To install the project, use the following command:

```bash
pip install -e .
```

## Installing  Dependencies
To install dependencies listed in the requirements.txt file, use the following 
command:

```bash
pip install -r requirements.txt
```

