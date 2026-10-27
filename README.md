🌦️ BBQ Weather Prediction Using Deep Learning

A binary classification project built using TensorFlow and Keras to predict outcomes from the BBQ Weather dataset. The project focuses on reducing overfitting through regularization techniques, validation monitoring, and neural network architecture optimization.

The final model achieved 96.19% accuracy and a 0.95 macro F1-score, demonstrating strong classification performance.

🎯 Project Objective

The primary objective is to develop a reliable neural network classifier that minimizes overfitting and generalizes effectively to unseen data. The project explores practical techniques for improving model stability and performance on the BBQ Weather dataset.

📂 Dataset Information

Dataset: BBQ Weather

Dataset Shape: (2923, 12)

Number of Samples: 2,923

Number of Input Features: 12

Task: Binary Classification

🧠 Model Architecture

The neural network was implemented using the Keras Sequential API, with ReLU activations in the hidden layers and a sigmoid activation in the output layer.

model = Sequential()

model.add(Dense(
    128,
    activation='relu',
    input_dim=12,
    kernel_regularizer=regularizers.l2(0.001)
))
model.add(Dropout(0.5))

model.add(Dense(
    64,
    activation='relu',
    kernel_regularizer=regularizers.l2(0.001)
))
model.add(Dropout(0.2))

model.add(Dense(
    32,
    activation='relu',
    kernel_regularizer=regularizers.l2(0.001)
))
model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))

Architecture Overview

Layer

Configuration

Input

12 features

Hidden Layer 1

128 neurons, ReLU

Dropout 1

50%

Hidden Layer 2

64 neurons, ReLU

Dropout 2

20%

Hidden Layer 3

32 neurons, ReLU

Dropout 3

20%

Output

1 neuron, Sigmoid

L2 regularization with a coefficient of 0.001 is applied to the hidden layers' weights.

⚠️ Challenge: Overfitting

During the initial training stages, the model exhibited signs of overfitting, including fluctuations in validation loss and a gap between training and validation performance.

To address these issues, several regularization and training strategies were explored.

🛡️ Techniques for Reducing Overfitting

1. Dropout Regularization

Dropout randomly disables a proportion of neurons during training, reducing the model's dependence on specific neurons and encouraging it to learn more robust patterns.

Dropout rates used:

After the 128-neuron layer: 0.5

After the 64-neuron layer: 0.2

After the 32-neuron layer: 0.2

2. L2 Regularization

L2 regularization penalizes large weights by adding a regularization term to the loss function.

[
L_{\text{total}} = L_{\text{original}} + \lambda \sum_i w_i^2
]

Where:

(\lambda) represents the regularization strength.

(w_i) represents the model weights.

A regularization coefficient of 0.001 was used to discourage excessively large weights and help control model complexity.

3. Validation Monitoring

A validation split of 0.2 was used to monitor the model's performance during training.

80% of the data was allocated for training.

20% was allocated for validation.

Validation metrics helped identify potential overfitting and assess how well the model generalized beyond its training data.

4. Batch Size Optimization

Different batch sizes were explored to improve training stability. Adjusting the batch size helped investigate gradient behavior, convergence, and validation-loss fluctuations.

📈 Training Analysis

The final training observations indicated:

A consistent decrease in training loss.

An improving validation-loss trend.

A relatively small gap between training and validation performance.

Stable optimization and promising generalization.

These observations suggest that the regularization strategies helped control overfitting. However, performance on a separate test set is needed to confirm generalization to truly unseen data.

📊 Model Performance

Classification Report

Class

Precision

Recall

F1-Score

0

0.98

0.98

0.98

1

0.92

0.92

0.92

Overall Metrics

Metric

Score

Accuracy

96.19%

Macro F1-Score

0.95

Weighted F1-Score

0.96

The results indicate strong classification performance across both classes, with an overall accuracy of 96.19%.

🛠️ Technologies Used

Python — Core programming language

TensorFlow — Deep learning framework

Keras — Neural network implementation

NumPy — Numerical computations

Pandas — Data manipulation and preprocessing

Matplotlib — Training visualization and plotting

Scikit-learn — Data preprocessing and model evaluation

💡 Key Learnings

This project provided practical experience with:

Understanding and identifying neural network overfitting.

Applying Dropout and L2 regularization.

Designing multilayer neural network architectures.

Monitoring training and validation performance.

Exploring the effect of batch size on optimization.

Evaluating classification models using precision, recall, F1-score, and accuracy.

Interpreting class-wise performance using a classification report.

🚀 Future Improvements

Potential enhancements include:

Early Stopping: Stop training when validation performance stops improving.

Learning Rate Scheduling: Adjust the learning rate during training.

Cross-Validation: Evaluate performance across multiple data splits.

Hyperparameter Tuning: Optimize layer sizes, dropout rates, and regularization strength.

Class Imbalance Handling: Investigate SMOTE or class weighting if the dataset is imbalanced.

ROC-AUC Analysis: Evaluate the model's ability to distinguish between the two classes.

Independent Test Evaluation: Assess performance on data not used during training or model selection.

📝 Conclusion

This project demonstrates the practical application of deep learning techniques for binary classification using the BBQ Weather dataset. By incorporating Dropout, L2 regularization, validation monitoring, and batch-size experimentation, the model achieved 96.19% accuracy with a macro F1-score of 0.95.

The project highlights the importance of balancing model complexity and predictive performance to develop neural networks that generalize effectively.

# 👨‍💻 Author

Rishav Kumar
ECE Undergraduate — NIT Silchar
