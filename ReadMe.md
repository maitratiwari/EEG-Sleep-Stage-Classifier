# Sleep Stage Classification using EEG Data 🧠💤

Automated sleep stage classification is a critical task in polysomnography. This repository provides a machine learning pipeline to classify sleep stages using Electroencephalography (EEG) signals from the **Sleep Physionet Dataset**.

## 🚀 Project Overview

The project utilizes the `MNE-Python` library for neurophysiological data processing and `Scikit-Learn` for classification. It explores two primary classification scenarios:
1. **Multi-Subject Classification (3 Stages):** Classification into Wake (W), NREM (Stages 1-4 combined), and REM (R) across 10 participants.
2. **Binary State Detection:** Distinguishing between 'Awake' and 'Sleeping' states.

## 📂 Repository Structure

* `EEG_Sleep_Classification_Multiple_Subject.ipynb`: A multi-subject study performing 3-stage classification (Wake, NREM, REM) via Leave-One-Group-Out (LOGO) cross-validation across 10 subjects to evaluate generalization to unseen participants.
* `EEG_Sleep_Classification_Two_Subject.ipynb`: A binary classification study (Awake vs. Sleeping) evaluating model transferability by training on one participant and testing on another to assess cross-subject performance in simplified states.

## 📊 Dataset
The models are trained on the **Physionet Sleep-EDF (Age)** dataset.
- **Format:** EDF (European Data Format).
- **Channels Used:** 2 EEG channels (Fpz-Cz, Pz-Oz).
- **Epoch Duration:** 30 seconds (standard clinical window).

## 🛠️ Technical Methodology

### 1. Preprocessing
- **Epoching:** Continuous EEG data is segmented into 30-second epochs.
- **Label Mapping:** - 3-Class: `W`, `N1/N2/N3/N4`, `REM`.
  - Binary: `Awake`, `Sleeping`.

### 2. Feature Extraction
Relative spectral power features are extracted from five standard EEG frequency bands:
- **Delta** (0.5–4.5 Hz) | **Theta** (4.5–8.5 Hz) | **Alpha** (8.5–11.5 Hz) 
- **Sigma** (11.5–15.5 Hz) | **Beta** (15.5–30.0 Hz)

Power Spectral Density (PSD) is calculated for each epoch, resulting in a feature vector representing the distribution of signal power across frequencies.

### 3. Classification
- **Model:** `RandomForestClassifier` (100 estimators).
- **Validation:** - **Leave-One-Group-Out (LOGO):** A robust cross-validation scheme where the model is trained on $N-1$ participants and tested on the held-out participant. This ensures the model generalizes to new subjects.

### 📊 Performance Evaluation

The models generate comprehensive performance metrics for each subject:
* **Confusion Matrices:** Visualizing misclassification between similar stages (e.g., N1 and REM).
* **Accuracy & F1-Scores:** Measuring the precision/recall balance for imbalanced sleep stages.
* **Subject-Wise Reports:** Tracking how biological variability impacts classification accuracy.

```bash
pip install mne numpy matplotlib scikit-learn