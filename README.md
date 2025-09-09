# SolarGuard — Solar Panel Defect Classification (Task-1)

**Goal.** Classifying Solar panel images into 6 classes: **BirdDrop, Clean, Dusty, ElectricalDamage, PhysicalDamage, SnowCovered** 

## Data & Splits
- Images supplied as class-wise folders; cleaned (removed junk/unreadable, fixed nested dirs).
- Final counts: BirdDrop 207, Clean 193, Dusty 190, ElectricalDamage 103, PhysicalDamage 69, SnowCovered 123 (total 885).
- Split rule from evaluator: **overall 80/20**, with **10% of train as validation** → ~**72/8/20** per class (stratified).

## Method (Why these choices)
- **Backbone:** MobileNetV3-Small (fast on CPU, ~1.5M trainable; easy to demo).  
- **Input:** 224×224, ImageNet mean/std (stable transfer learning).  
- **Augment (light):** flips/rotations/brightness (avoid hallucinating defects).  
- **Imbalance:** class-weighted CrossEntropy (weights from inverse frequency).  
- **Optim:** AdamW (lr=3e-4, wd=1e-4); **StepLR(7, γ=0.5)**.  
- **Early stopping:** on **macro-F1** (treats all classes equally; important for minority classes).

## Results (Test set)
- **Accuracy:** 0.8547  
- **Macro-F1:** 0.8491  
- **Weighted-F1:** 0.8544  
- Confusion matrix saved at: `reports/figures/confusion_matrix_test.png`  
- Notes: “SnowCovered” is easiest; common confusions: **Dusty ↔ BirdDrop/Clean**.

## Repo Structure
