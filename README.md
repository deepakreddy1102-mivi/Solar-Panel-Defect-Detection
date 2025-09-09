# SolarGuard — Solar Panel Defect Classification (Task-1) ----> [My First repository]

**Goal.** Take a panel photo and predict one of six classes:
**BirdDrop, Clean, Dusty, ElectricalDamage, PhysicalDamage, SnowCovered**.
This is meant to speed up O&M triage (cleaning vs. technician visit).

---

## Data & Splits
- Images were provided as class-wise folders. I removed junk/unreadable files and fixed nested dirs (e.g., `BirdDrop/New → BirdDrop`).
- Final counts after cleanup (total **885**):
  - BirdDrop **207**, Clean **193**, Dusty **190**, ElectricalDamage **103**, PhysicalDamage **69**, SnowCovered **123**.
- Per evaluator’s instruction: **overall 80/20** split. From **train**, I reserved **10%** for validation → roughly **72/8/20** per class (stratified).

---

## Method — why I chose this setup
- **Backbone:** **MobileNetV3-Small** – tiny (~1.5M trainable), fast on CPU, good enough for a live demo.
- **Input size & stats:** **224×224**, ImageNet mean/std – standard transfer setup for stable fine-tuning.
- **Augmentation (light):** horizontal flip, small rotations, mild brightness change – adds variety without “inventing” defects.
- **Imbalance handling:** class-weighted CrossEntropy (weights from inverse class frequency).
- **Optimiser:** **AdamW** (lr=3e-4, wd=1e-4).
- **Scheduler:** **StepLR(step=7, gamma=0.5)**.
- **Early stopping metric:** **macro-F1** (treats all classes equally; protects minority classes like Physical/Electrical damage).

---

## Results (test set)
- **Accuracy:** **0.8547**
- **Macro-F1:** **0.8491**
- **Weighted-F1:** **0.8544**
- Confusion matrix: `reports/figures/confusion_matrix_test.png`
- Quick read: **SnowCovered** is the easiest. Most mistakes are **Dusty ↔ BirdDrop/Clean** (texture + lighting).

---

## Repo structure
