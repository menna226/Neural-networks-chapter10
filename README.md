# Chapter 10 – Neural Networks

This repository contains two neural network projects built with TensorFlow/Keras as part of Chapter 10 exercises from the book **[Grokking Machine Learning](https://www.manning.com/books/grokking-machine-learning)** by **Luis Serrano** (Manning Publications).

> 📖 Original author's repository: [github.com/luisguiserrano/manning](https://github.com/luisguiserrano/manning)

---

## 📁 Repository Structure

```
Neural-networks-chapter10/
│
├── data/
│   └── Hyderabad.csv               # Housing dataset from Kaggle
│
├── src/
│   ├── house_price_predictions.py  # Housing price prediction model
│   └── image_recognition.py        # MNIST digit recognition model
│
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🏠 Project 1 – Housing Price Prediction (Hyderabad)

### Project Overview

This project predicts house prices in Hyderabad, India, using a neural network (deep learning) model. The dataset comes from Kaggle: **[Housing Prices in Metropolitan Areas of India](https://www.kaggle.com/datasets/ruchi798/housing-prices-in-metropolitan-areas-of-india)**.

- **Goal:** Predict the `Price` of houses based on their features (amenities, area, number of rooms, etc.).
- **Approach:** Use a feedforward neural network with fully connected layers (Dense) and dropout for regularization.
- **Metrics:** Root Mean Squared Error (RMSE) to measure how close predictions are to actual prices.

### Dataset Details

- CSV file: `Hyderabad.csv`
- Features include numeric columns for amenities, rooms, floor, etc.
- Target: `Price` (house price in Indian Rupees).
- **Special placeholder values:**

> **Note:** For some houses, nothing was mentioned about certain amenities, so `9` was used to mark missing values. This indicates absence of information, not necessarily absence of the feature in real life.

### Data Preprocessing

1. **Remove non-numeric columns:** `Location` column dropped for simplicity.
2. **Handle placeholders:** Replace `9` with `0` for missing amenity information.
3. **Train/test split:** 80% training, 20% testing.
4. **Feature scaling:** Standardize all input features using `StandardScaler`. Standardize the target (`Price`) to improve neural network training.
5. **Optional improvements:**
   - Log-transform target prices for high variance.
   - Encode `Location` to include geographic influence on price.

### Model Architecture

```
Input layer → 38 features
├── Dense 128 → ReLU + Dropout(0.2)
├── Dense 64  → ReLU + Dropout(0.2)
├── Dense 32  → ReLU + Dropout(0.2)
└── Output layer → 1 neuron (regression)
```

- **Optimizer:** Adam
- **Loss function:** Mean Squared Error (MSE)
- **Metric:** Root Mean Squared Error (RMSE)

### Training Details

- Epochs: 150
- Batch size: 32
- Validation split: 20% of training set for monitoring overfitting
- Random seed set for reproducibility (`np.random.seed(0)` and `tf.random.set_seed(1)`)

### Results

Example predictions (first 5 test samples):

| Prediction  | Actual    |
|-------------|-----------|
| 7,098,621   | 6,500,000 |
| 10,034,832  | 9,906,000 |
| 5,317,469   | 7,000,000 |
| 5,601,634   | 5,500,000 |
| 5,905,839   | 4,236,000 |

- **RMSE on test set:** ~ *(print your final test RMSE)*
- Predictions are in the correct range and reflect the patterns in the dataset.

### Key Improvements Over Basic Version

1. Replaced placeholder `9` with `0` to clean data.
2. Standardized both features and target to improve model convergence.
3. Added more layers (128 → 64 → 32) with dropout to reduce overfitting.
4. Increased epochs to 150 to improve learning.
5. Validation split included to monitor overfitting.

**Result:** Much closer predictions to actual prices compared to the basic version.

### Future Improvements

- Encode `Location` for better price prediction based on neighborhood.
- Use log-transform for prices to reduce the effect of high-value outliers.
- Tune hyperparameters (neurons, learning rate, batch size) for better accuracy.
- Add more features from dataset (e.g., total rooms, amenities count).

---

## 🔢 Project 2 – MNIST Image Recognition

### Project Overview

- Built a simple image recognition neural network using TensorFlow/Keras.
- Trained on the **MNIST dataset**: 60,000 training images, 10,000 test images of handwritten digits (0–9).

### Model Architecture

```
Input: 28×28 flattened → 784 features
├── Dense 128 → ReLU + Dropout(0.2)
├── Dense 64  → ReLU + Dropout(0.2)
└── Output: 10 neurons → Softmax
```

### Training Notes

- **Loss function:** Categorical Crossentropy
- **Optimizer:** Adam
- **Epochs:** 10
- **Batch size:** 10
- **Accuracy on test set:** ~93–94% (varies due to randomness)

### Why Results Differ from Tutorials

Even with the same code, your results may differ slightly from the author's:

1. **Random weight initialization** → network starts from slightly different points.
2. **Dropout layers** → randomly ignore 20% of neurons each batch → small changes in learned weights.
3. **Data shuffling** → batches are shuffled every epoch → slightly different gradient updates.
4. **GPU vs CPU computation** → floating-point precision can differ slightly.

> **Key idea:** Neural networks are stochastic, so minor differences in predictions and accuracy are completely normal.

### Common Errors and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `ValueError: cannot reshape array of size ...` | Shape mismatch | Ensure `x_train.reshape(-1, 28*28)` |
| Predictions off by 1 or 2 digits | Randomness in training / dropout | This is normal; predictions vary slightly each run |
| Plotting wrong image | Index mismatch | Double-check the index when plotting `x_train[i]` or `x_test[i]` |

---

## ⚙️ Requirements

```
tensorflow
numpy
pandas
scikit-learn
matplotlib
```

Install with:

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib
```

## 🚀 How to Run

```bash
# Housing price prediction
python src/house_price_predictions.py

# MNIST image recognition
python src/image_recognition.py
```

---

## 📚 References

- **Book:** [Grokking Machine Learning](https://www.manning.com/books/grokking-machine-learning) — Luis Serrano, Manning Publications
- **Author's GitHub:** [github.com/luisguiserrano/manning](https://github.com/luisguiserrano/manning)
- **Dataset:** [Housing Prices in Metropolitan Areas of India](https://www.kaggle.com/datasets/ruchi798/housing-prices-in-metropolitan-areas-of-india) — Kaggle (ruchi798)
- **MNIST Dataset:** Loaded directly via `tensorflow.keras.datasets.mnist`

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
