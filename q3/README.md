# Perceptron From Scratch

This project implements a single-neuron logistic regression model (perceptron) to classify fruits (apples vs bananas) based on their features.

## Project Structure

- `fruit.csv` - Dataset containing fruit features:
  - length_cm: Fruit length in centimeters
  - weight_g: Fruit weight in grams
  - yellow_score: Color score from 0-1 (0 = not yellow, 1 = fully yellow)
  - label: 0 for apple, 1 for banana

- `perceptron.ipynb` - Jupyter notebook with the implementation
- `perceptron_script.py` - Python script version of the notebook
- `reflection.md` - Reflection on the implementation and learning outcomes

## How to Run

### Using Python Script

```bash
python perceptron_script.py
```

This will:
1. Load and analyze the fruit dataset
2. Train a logistic regression model from scratch
3. Compare different learning rates
4. Generate visualization plots saved as PNG files

### Output Files

The script generates the following visualization files:
- `training_progress.png` - Loss and accuracy over epochs
- `learning_rate_comparison.png` - Individual learning rate performance
- `learning_rate_comparison_combined.png` - Combined learning rate comparison
- `decision_boundary.png` - Visualization of the decision boundary

## Implementation Details

- The model uses a single neuron with sigmoid activation
- Training is done using batch gradient descent
- The loss function is binary cross-entropy
- Early stopping is implemented when loss < 0.05
- The model is evaluated using accuracy

## Learning Outcomes

This project demonstrates:
1. How to implement a basic neural network building block from scratch
2. Understanding of gradient descent optimization
3. The impact of learning rate on model convergence
4. Techniques for visualizing model training progress and decision boundaries 