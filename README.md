# WiDS-5.0-End-to-End ML Systems on new datasets: From Transfer Learning to Production Deployment with Federated Averaging
Project ID : 80
Name : Vanshika Nalamasa
Mentors : Yuvraj Parekh
          Chaitanya Deshkar

# Plant Disease Classification Using Machine Learning

**WiDS 5.0 Workshop Project - Midterm Checkpoint**  
*Analytics Club, IIT Bombay*

---

## Project Overview

This project aims to develop an automated plant disease classification system using machine learning techniques. The system analyzes images of plant leaves to identify various diseases, which can assist farmers and agricultural experts in early disease detection and crop management.

### Problem Statement

Plant diseases significantly impact agricultural productivity and food security. Early and accurate disease detection is crucial for effective treatment and crop protection. This project leverages computer vision and machine learning to automate the disease identification process from leaf images.

### Dataset

- **Source:** PlantVillage Dataset (Kaggle)
- **Size:** 54,305 images
- **Classes:** 38 disease categories across multiple plant species
- **Format:** RGB images (varying dimensions)
- **Plants Covered:** Tomato, Potato, Pepper, Apple, Cherry, Grape, and others

---

## Repository Structure

```
plant-disease-classification/
│
├── notebooks/
│   ├── week1_eda.ipynb                    # Exploratory Data Analysis
│   ├── week2_classical_ml_baseline.ipynb  # Classical ML Experiments
│   └── week3_deep_learning.ipynb          # CNN & Transfer Learning
│
│
└── README.md                              # Project documentation
```

---

## Methodology

### Phase 1: Exploratory Data Analysis (Week 1)

#### Objectives
- Understand dataset characteristics and distribution
- Identify potential challenges for model training
- Analyze visual patterns in diseased vs healthy leaves
- Assess class balance and data quality

#### Key Findings

**1. Class Distribution Analysis**
- Total images: 54,305
- Classes: 38 distinct disease categories
- Imbalance ratio: ~3:1 (max to min samples per class)
- Dominant class: Tomato-related diseases (14 classes)

**2. Image Properties**
- Average dimensions: 256×256 pixels
- Aspect ratio: Consistent (~1.0)
- Color space: RGB with high green channel dominance
- File format: JPEG with average size 45KB

**3. Visual Pattern Analysis**
- Diseased leaves exhibit higher texture complexity
- Edge density is significantly greater in diseased samples
- Color variation between disease classes is moderate
- Healthy leaves show uniform texture and consistent coloring

**4. Predicted Challenges**
- High visual similarity between certain disease pairs (e.g., Early Blight vs Late Blight)
- Class imbalance may lead to bias toward majority classes
- Intra-class variation due to different imaging conditions
- Fine-grained classification required for similar symptoms

#### Deliverables
- Comprehensive EDA notebook with 12 analysis sections
- 15+ visualizations including distribution plots, sample grids, and statistical charts
- Documentation of insights linking to model development strategy

---

### Phase 2: Classical Machine Learning Baseline (Week 2)

#### Objectives
- Establish performance baseline using traditional ML algorithms
- Understand limitations of non-deep learning approaches
- Quantify the performance gap that deep learning must address

#### Preprocessing Pipeline

```
Raw Image (H × W × 3)
    ↓
Resize to 32×32
    ↓
Flatten to 1D vector (3,072 features)
    ↓
StandardScaler normalization
    ↓
Train/Test Split (80/20)
```

**Rationale for 32×32 resolution:** Balance between computational efficiency and information retention for rapid baseline experimentation.

#### Models Evaluated

| Model | Configuration | Accuracy | Training Time |
|-------|--------------|----------|---------------|
| Dummy Classifier | most_frequent | 2.63% | <1s |
| K-Nearest Neighbors | k=5, distance-weighted | 44.82% | 5s |
| Logistic Regression | multinomial, L2 reg | 47.54% | 8s |
| Linear SVM | C=1.0, linear kernel | 55.26% | 15s |
| **Random Forest** | **100 trees, max_depth=20** | **65.8%** | **10s** |

#### Results Analysis

**Best Performing Model: Random Forest (65.8% accuracy)**

**Per-Class Performance (Top 10 Classes):**
- Tomato Healthy: 83.3%
- Tomato Mosaic Virus: 80.0%
- Tomato Bacterial Spot: 73.3%
- Tomato Late Blight: 26.7%
- Tomato Septoria Leaf Spot: 13.3%
- Tomato Early Blight: 10.0%

**Key Observations:**
1. Strong performance on visually distinct classes (healthy vs diseased)
2. Significant confusion between visually similar diseases
3. Performance deteriorates with increased inter-class similarity
4. Model struggles with fine-grained classification tasks

#### Limitations of Classical ML Approach

**1. Spatial Information Loss**
- Flattening operation destroys 2D structure
- Adjacent pixels in image space become arbitrary positions in feature vector
- No understanding of local spatial relationships

**2. Feature Engineering Constraints**
- Raw pixel values lack semantic meaning
- No hierarchical feature representation
- Cannot learn abstract visual concepts (edges, textures, patterns)

**3. Scalability Issues**
- High-dimensional feature space (3,072 features for 32×32 images)
- Computational complexity increases with image resolution
- Limited capacity for complex decision boundaries

**Conclusion:** Classical ML achieves 65.8% accuracy, establishing a baseline that demonstrates the necessity for deep learning approaches capable of hierarchical feature extraction.

---

## Technical Implementation

### Environment Setup

**Platform:** Kaggle Notebooks (GPU P100)

**Dependencies:**
```python
# Data & Visualization
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.1
seaborn==0.12.2
Pillow==10.0.0

# Classical ML
scikit-learn==1.3.0

# Deep Learning
tensorflow==2.15.0
keras==2.15.0
```

### Data Loading Configuration

```python
IMG_SIZE = (32, 32)
MAX_SAMPLES_PER_CLASS = 150
TEST_SIZE = 0.2
RANDOM_STATE = 42
```

### Model Hyperparameters

**Random Forest (Best Model):**
- `n_estimators`: 100
- `max_depth`: 20
- `random_state`: 42
- `n_jobs`: -1

**Preprocessing:**
- Scaling: StandardScaler (μ=0, σ=1)
- Train-test stratification: Maintained class distribution

---

## Results Summary

### Quantitative Metrics

| Metric | Value |
|--------|-------|
| Baseline Floor (Dummy) | 2.63% |
| Classical ML Ceiling (RF) | 65.8% |
| Improvement over Baseline | +63.17% |
| Classes with >70% Accuracy | 8/38 |
| Classes with <30% Accuracy | 12/38 |

### Confusion Matrix Analysis (Week 3)

The confusion matrix for the transfer learning model (83.2% accuracy) reveals significant improvements:

**Improvements from Week 2:**
- Better classification of similar disease pairs
- Reduced confusion between Early Blight and Late Blight
- Higher confidence in disease-specific patterns
- More consistent performance across plant species

**Remaining Challenges:**
- Some confusion persists between visually similar diseases
- Minority classes with limited samples show lower accuracy
- Image quality and lighting variations still affect predictions

**Compared to Week 2:**
- Week 2 worst performers: Early Blight (10%), Late Blight (26.7%), Septoria (13.3%)
- Week 3 shows expected improvement in these challenging classes through learned spatial features

### Key Improvements from Classical ML

**Feature Learning Hierarchy:**
```
Layer 1: Edge detection (horizontal, vertical, diagonal)
Layer 2: Texture recognition (leaf veins, spot patterns)
Layer 3: Complex patterns (lesion shapes, disease signatures)
Layer 4-5: Disease classification
```

**Advantages of CNN Approach:**
1. **Spatial Preservation**: No flattening; maintains 2D structure
2. **Translation Invariance**: Recognizes patterns regardless of position
3. **Hierarchical Features**: Learns from simple to complex patterns
4. **Transfer Learning**: Leverages knowledge from millions of images

**Limitations Addressed:**
- Classical ML: 65.8% (flattened vectors, no spatial understanding)
- Simple CNN: 68.3% (learns spatial patterns but limited depth)
- Transfer Learning: 83.2% (pre-trained features + domain adaptation)

**Why Transfer Learning Won:**
- MobileNetV2 trained on ImageNet already learned universal visual features
- Edges, textures, and shapes learned from 1.4M images transfer to plant leaves
- Only disease-specific patterns needed to be learned
- Result: 17.4% improvement over classical ML with similar training time

---

## Results Summary (Updated)

### Quantitative Metrics

| Metric | Week 2 (Classical) | Week 3 (Deep Learning) | Improvement |
|--------|-------------------|----------------------|-------------|
| Best Accuracy | 65.8% (RF) | **83.2%** (Transfer) | **+17.4%** |
| Training Time | ~10 seconds | ~20 minutes (GPU) | - |
| Feature Engineering | Manual (flattening) | Automatic (learned) | ✅ |
| Spatial Understanding |  Lost in flattening | Preserved | ✅ |
| Parameters | N/A | ~2.8M (500K trainable) | - |

### Model Progression

```
Week 2: Random Forest        → 65.8%
Week 3: Simple CNN           → 68.3% (+2.5%)
Week 3: Transfer (Frozen)    → 83.2% (+17.4%) ⭐ BEST
Week 3: Transfer (Fine-tuned)→ 76.3% (+10.5%)
```

**Key Insight:** Frozen transfer learning outperformed fine-tuning, suggesting:
- Pre-trained features were already optimal for this task
- Limited dataset (200 samples/class) makes fine-tuning prone to overfitting
- For small datasets, frozen transfer learning is often the best strategy

### Confusion Matrix Comparison

**Week 2 Problem Classes:**
- Tomato Early Blight: 10% accuracy
- Tomato Late Blight: 26.7% accuracy
- Tomato Septoria: 13.3% accuracy

**Week 3 Expected Improvements:** 
The CNN's spatial feature learning should improve classification of these challenging disease pairs. The confusion matrix (see notebook visualization) shows reduced off-diagonal confusion compared to Week 2's classical ML results.

---

## Deep Learning Insights

### Why CNNs Outperform Classical ML

**1. Spatial Relationships**
- Classical ML: Treats pixels 100 and 101 as independent features
- CNN: Understands these pixels are neighbors in the image
- **Impact on accuracy: +17.4% improvement (65.8% → 83.2%)**

**2. Feature Hierarchy**
```
Input Image (128×128×3)
    ↓
Conv Layer 1: Learn 32 edge detectors
    → Detects: vertical lines, horizontal lines, curves
    → Similar to Gabor filters but learned automatically
    ↓
Conv Layer 2: Combine edges → 64 texture patterns
    → Detects: leaf veins, color gradients, spots
    → Learns "circular lesion" = edges arranged in circles
    ↓
Conv Layer 3: Combine textures → 128 complex patterns
    → Detects: lesion shapes, disease signatures
    → Learns "late blight" = specific texture + color + shape
    ↓
Classification: Map patterns to diseases
    → Combines all learned features for final prediction
```

**3. Transfer Learning Advantage**
- MobileNetV2 pre-trained on ImageNet (1.4M images, 1000 classes)
- Already learned: edges, textures, shapes, objects
- Fine-tuning: Teaches "which patterns indicate which plant diseases"
- Result: **83.2% accuracy vs 68.3% training from scratch**

### Experimental Results Analysis

| Experiment | Result | Insight |
|-----------|--------|---------|
| Simple CNN | 68.3% | Spatial features help but limited depth constrains learning |
| Transfer (Frozen) | 83.2% | Pre-trained features transfer remarkably well |
| Transfer (Fine-tuned) | 76.3%  | Overfitting occurred; dataset too small for safe fine-tuning |

**Unexpected Finding:** Fine-tuning decreased performance by 6.9%. This counter-intuitive result teaches an important lesson: **more training is not always better**. With limited data, frozen transfer learning often outperforms fine-tuning because it avoids overfitting on the small target dataset.

### What Makes Transfer Learning Work

**Pre-learned Low-Level Features:**
- Horizontal/vertical edges (learned from ImageNet buildings, objects)
- Color gradients (learned from natural images)
- Texture patterns (learned from animal fur, tree bark, etc.)

**Domain Transfer:**
These features, though learned from general images, apply perfectly to plant leaves:
- Building edges → Leaf edges
- Animal textures → Disease spot textures
- Object shapes → Lesion shapes

**Efficiency:**
- Training from scratch: 10 epochs to reach 68.3%
- Transfer learning: 5 epochs to reach 83.2%
- Transfer learning trains **faster** and performs **better**

### Training Observations

**Simple CNN (68.3%):**
- Showed signs of overfitting after 7-8 epochs
- Training accuracy continued rising while validation plateaued
- Dropout layers (0.25, 0.5) helped but were insufficient alone
- Data augmentation was crucial for regularization
- Limited by shallow architecture and small dataset

**Transfer Learning - Frozen (83.2%):**
- Rapid convergence within 5-6 epochs
- Minimal overfitting due to pre-trained feature quality
- Training and validation curves remained close
- Pre-trained features generalized excellently to plant diseases
- Best performance with least training time

**Transfer Learning - Fine-tuned (76.3%):**
- Performance degraded from 83.2% to 76.3%
- Training accuracy increased but validation decreased
- Clear indication of overfitting during fine-tuning
- Small dataset (200 samples/class) insufficient for safe fine-tuning
- Learning rate 0.0001 may still have been too aggressive

**Key Learning:** For datasets with <500 samples per class, frozen transfer learning often outperforms fine-tuning. The pre-learned features from ImageNet are already sophisticated enough that updating them risks overfitting on the smaller target dataset.

---

## Phase 3: Deep Learning Implementation (Week 3)

### Objectives
- Develop Convolutional Neural Networks (CNNs) for hierarchical feature learning
- Apply transfer learning with pre-trained models
- Achieve significant improvement over classical ML baseline (65.8%)

### Architecture Development

#### Model 1: Simple CNN (From Scratch)

**Architecture:**
```
Conv2D(32) -> MaxPool -> Dropout(0.25)
Conv2D(64) -> MaxPool -> Dropout(0.25)
Conv2D(128) -> MaxPool -> Dropout(0.25)
Flatten -> Dense(256) -> Dropout(0.5) -> Dense(38, softmax)
```

**Configuration:**
- Input size: 128×128×3
- Total parameters: ~1.2M
- Optimizer: Adam
- Loss: Categorical crossentropy
- Training epochs: 10 (with early stopping)

**Results:**
- Validation accuracy: 68.3
- Training time: ~15 minutes (GPU)

#### Model 2: Transfer Learning (MobileNetV2)

**Architecture:**
```
MobileNetV2 (frozen, ImageNet weights)
GlobalAveragePooling2D
Dropout(0.5)
Dense(256, ReLU) -> Dropout(0.3)
Dense(38, softmax)
```

**Configuration:**
- Base model: MobileNetV2 (pre-trained on ImageNet)
- Trainable parameters: ~500K (head only)
- Total parameters: ~2.8M
- Fine-tuning strategy: Last 20 layers unfrozen
- Learning rates:
  - Initial training: 0.001
  - Fine-tuning: 0.0001

**Results:**
- Validation accuracy (frozen): 83.25
- Validation accuracy (fine-tuned): 76.29
- Training time: ~20 minutes (GPU)

### Data Augmentation Strategy

```python
augmentation_pipeline = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    fill_mode='nearest'
)
```

**Rationale:**
- **Rotation**: Plant leaves can be photographed from any angle
- **Shifts**: Disease symptoms appear at various positions
- **Flips**: Bilateral symmetry in leaf structure
- **Zoom**: Simulates different camera distances

### Training Strategy

**Phase 1: Initial Training (Frozen Base)**
- Epochs: 10
- Batch size: 32
- Callbacks: Early stopping (patience=3), ReduceLROnPlateau

**Phase 2: Fine-tuning (Unfrozen Layers)**
- Epochs: 5
- Learning rate: 0.0001 (10× reduction)
- Unfrozen layers: Last 20 layers of MobileNetV2

### Performance Metrics

| Model | Validation Accuracy | Improvement over Baseline |
|-------|-------------------|--------------------------|
| Week 2 Baseline (RF) | 65.8% | - |
| Simple CNN | 68.3 | 3.5 |
| Transfer Learning | 83.2 | 17.4  |
| Fine-tuned Model | 76.3 | 10.5 |

### Success Criteria

- Minimum threshold: 73% (baseline + 7%)
- Target accuracy: 85%
- Expected improvement: +15-20% over classical ML (65.8%)
- Per-class accuracy: >70% for majority classes

---

## Future Work (Week 4)

### Proposed Enhancements

**Model Optimization:**
- Experiment with higher resolution (224×224 to capture finer details)
- Try alternative architectures (ResNet50, EfficientNet-B0)
- Implement ensemble methods (voting across multiple models)
- Add attention mechanisms to focus on disease regions
- Address fine-tuning overfitting with different strategies:
  - Smaller learning rate (1e-5 instead of 1e-4)
  - Fewer unfrozen layers (10 instead of 20)
  - More conservative fine-tuning epochs

**Data Improvements:**
- Increase samples per class from 200 to full dataset
- Implement class-weighted loss for imbalanced classes
- More aggressive augmentation (mixup, cutout)
- Collect additional data for minority classes

**Deployment Strategy:**
- Model quantization for mobile deployment (TFLite)
- REST API development using Flask/FastAPI
- Web interface for farmer-friendly usage
- Real-time inference optimization

**Interpretability:**
- Implement Grad-CAM for model interpretability
- Visualize what features each layer learns
- Analyze failure cases systematically
- Create decision explanation interface

### Lessons for Future Projects

1. **Start with transfer learning** - Don't waste time training from scratch
2. **Be cautious with fine-tuning** - Only when dataset is large enough (>1000 samples/class)
3. **Data augmentation is crucial** - Especially for small datasets
4. **Monitor validation closely** - Training accuracy alone is misleading
5. **Simple often wins** - Frozen transfer learning beat fine-tuned version


## Key Learnings

### Technical Insights

1. **Importance of Baseline Establishment**
   - Multiple baseline models provide robust performance benchmarks
   - Dummy classifier defines absolute minimum performance threshold
   - Classical ML ceiling (65.8%) quantifies the deep learning performance gap

2. **Feature Representation Matters**
   - Raw pixel values insufficient for complex visual tasks
   - Spatial relationship preservation crucial for image classification
   - Hierarchical feature learning necessary for fine-grained classification
   - **Result:** 17.4% improvement from spatial feature learning

3. **Class Imbalance Impact**
   - Moderate imbalance (3:1) affects minority class performance
   - Data augmentation partially mitigates this issue
   - Transfer learning helps by providing robust pre-trained features

4. **Transfer Learning vs Training from Scratch**
   - Frozen transfer learning: 83.2% (best)
   - Simple CNN from scratch: 68.3%
   - **Lesson:** Pre-trained models provide massive advantage
   - For small datasets (<500 samples/class), frozen weights often optimal

5. **The Fine-tuning Paradox**
   - Fine-tuning decreased accuracy from 83.2% to 76.3%
   - **Cause:** Overfitting on small dataset (200 samples/class)
   - **Solution:** Keep base frozen or use very conservative learning rates
   - **Rule of thumb:** Fine-tune only with >1000 samples per class

### Methodological Insights

1. **EDA-Driven Model Development**
   - Visual pattern analysis predicts model challenges
   - Edge detection preview reveals feature importance
   - Color distribution analysis informs feature engineering
   - **Validation:** Week 1 predictions of confusion pairs confirmed in Week 2-3 results

2. **Iterative Experimentation**
   - Small image sizes (128×128) enable rapid prototyping
   - Multiple model comparison validates findings
   - Systematic evaluation guides architecture decisions
   - **Progression:** 65.8% → 68.3% → 83.2% shows clear improvement path

3. **The Power of Pre-training**
   - Transfer learning achieved 83.2% vs 68.3% from scratch
   - **14.9% gap** demonstrates value of ImageNet pre-training
   - Frozen features sufficient for many tasks
   - Training time similar but results dramatically different

4. **Understanding Model Failure Modes**
   - Simple CNN overfits despite regularization
   - Fine-tuning overfits on small datasets
   - Data augmentation helps but has limits
   - **Lesson:** Know when more complexity hurts rather than helps


## Project Outcomes

### Quantitative Achievements

| Milestone | Target | Achieved 
|-----------|--------|----------
| Week 1: Complete EDA
| Week 2: Classical Baseline | >50% | 65.8% 
| Week 3: Beat baseline by 15% | 73%+ | 83.2% 
| Week 3: Target accuracy | 85% | 83.2% 
| Overall improvement | +20% | +17.4% 







## References

1. PlantVillage Dataset: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset



