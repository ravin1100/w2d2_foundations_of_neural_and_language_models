import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load the fruit dataset
print("Loading the fruit dataset...")
df = pd.read_csv('fruit.csv')
print(df.head())

# Display basic statistics
print("\nDataset Statistics:")
print(df.describe())

# Check for class balance
print("\nClass Distribution:")
print(df['label'].value_counts())

# Prepare the data
X = df[['length_cm', 'weight_g', 'yellow_score']].values
y = df['label'].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Original features (first 3 rows):")
print(X[:3])
print("\nScaled features (first 3 rows):")
print(X_scaled[:3])

class LogisticNeuron:
    def __init__(self, n_features):
        # Initialize weights and bias randomly
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = np.random.randn() * 0.01
        
    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-z))
    
    def forward(self, X):
        """Forward pass"""
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def compute_loss(self, y_pred, y_true):
        """Binary cross-entropy loss"""
        epsilon = 1e-15  # Small value to avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip predictions to avoid numerical issues
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def compute_gradients(self, X, y_true, y_pred):
        """Compute gradients for weights and bias"""
        m = X.shape[0]
        dw = (1/m) * np.dot(X.T, (y_pred - y_true))
        db = (1/m) * np.sum(y_pred - y_true)
        return dw, db
    
    def update_parameters(self, dw, db, learning_rate):
        """Update weights and bias using gradients"""
        self.weights -= learning_rate * dw
        self.bias -= learning_rate * db
    
    def train(self, X, y, learning_rate=0.1, epochs=1000, early_stop_loss=0.05):
        """Train the model using batch gradient descent"""
        losses = []
        accuracies = []
        
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)
            
            # Compute loss
            loss = self.compute_loss(y_pred, y)
            losses.append(loss)
            
            # Compute accuracy
            y_pred_binary = (y_pred >= 0.5).astype(int)
            accuracy = np.mean(y_pred_binary == y)
            accuracies.append(accuracy)
            
            # Compute gradients
            dw, db = self.compute_gradients(X, y, y_pred)
            
            # Update parameters
            self.update_parameters(dw, db, learning_rate)
            
            # Print progress every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            
            # Early stopping if loss is below threshold
            if loss < early_stop_loss:
                print(f"Early stopping at epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
                break
        
        return losses, accuracies

# Initialize the model
print("\nInitializing the model...")
np.random.seed(42)  # For reproducibility
model = LogisticNeuron(n_features=X_scaled.shape[1])

# Initial predictions before training
initial_predictions = model.forward(X_scaled)
initial_pred_binary = (initial_predictions >= 0.5).astype(int)
initial_accuracy = np.mean(initial_pred_binary == y)
initial_loss = model.compute_loss(initial_predictions, y)

print(f"Initial random model - Loss: {initial_loss:.4f}, Accuracy: {initial_accuracy:.4f}")
print("Initial weights:", model.weights)
print("Initial bias:", model.bias)

# Train the model
print("\nTraining the model...")
learning_rate = 0.1
epochs = 1000
early_stop_loss = 0.05

losses, accuracies = model.train(X_scaled, y, learning_rate, epochs, early_stop_loss)

# Final predictions after training
final_predictions = model.forward(X_scaled)
final_pred_binary = (final_predictions >= 0.5).astype(int)
final_accuracy = np.mean(final_pred_binary == y)
final_loss = model.compute_loss(final_predictions, y)

print(f"\nFinal model - Loss: {final_loss:.4f}, Accuracy: {final_accuracy:.4f}")
print("Final weights:", model.weights)
print("Final bias:", model.bias)

# Plot the training progress
print("\nPlotting training progress...")
plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Binary Cross-Entropy Loss')
plt.grid(True)

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(accuracies)
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)

plt.tight_layout()
plt.savefig('training_progress.png')
print("Training progress plot saved as 'training_progress.png'")

# Compare different learning rates
print("\nComparing different learning rates...")
learning_rates = [0.01, 0.1, 0.5, 1.0]
epochs = 500
results = []

plt.figure(figsize=(14, 10))

for i, lr in enumerate(learning_rates):
    # Reset model with same initialization
    np.random.seed(42)
    model_lr = LogisticNeuron(n_features=X_scaled.shape[1])
    
    # Train with this learning rate
    losses_lr, accuracies_lr = model_lr.train(X_scaled, y, learning_rate=lr, epochs=epochs, early_stop_loss=0.001)
    results.append((lr, losses_lr, accuracies_lr))
    
    # Plot loss
    plt.subplot(2, 2, i+1)
    plt.plot(losses_lr)
    plt.title(f'Loss with Learning Rate = {lr}')
    plt.xlabel('Epoch')
    plt.ylabel('Binary Cross-Entropy Loss')
    plt.grid(True)

plt.tight_layout()
plt.savefig('learning_rate_comparison.png')
print("Learning rate comparison plot saved as 'learning_rate_comparison.png'")

# Plot all learning rates on the same graph for comparison
plt.figure(figsize=(12, 5))

for lr, losses_lr, _ in results:
    plt.plot(losses_lr[:100], label=f'LR = {lr}')  # Show first 100 epochs for clarity

plt.title('Loss Comparison for Different Learning Rates')
plt.xlabel('Epoch')
plt.ylabel('Binary Cross-Entropy Loss')
plt.legend()
plt.grid(True)
plt.savefig('learning_rate_comparison_combined.png')
print("Combined learning rate comparison plot saved as 'learning_rate_comparison_combined.png'")

# Visualize the decision boundary
print("\nVisualizing decision boundary...")
plt.figure(figsize=(10, 8))

# We'll use length and yellow_score as our 2D features for visualization
feature1_idx = 0  # length_cm
feature3_idx = 2  # yellow_score

# Create a mesh grid
x_min, x_max = X_scaled[:, feature1_idx].min() - 1, X_scaled[:, feature1_idx].max() + 1
y_min, y_max = X_scaled[:, feature3_idx].min() - 1, X_scaled[:, feature3_idx].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# For each point in the mesh, calculate the prediction
Z = np.zeros((xx.shape[0], xx.shape[1]))
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        # Create a feature vector with the average weight_g (feature2) value
        avg_feature2 = np.mean(X_scaled[:, 1])
        features = np.array([xx[i, j], avg_feature2, yy[i, j]])
        Z[i, j] = model.forward(features)

# Plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
plt.colorbar()

# Plot the training points
scatter = plt.scatter(X_scaled[:, feature1_idx], X_scaled[:, feature3_idx], 
                     c=y, edgecolors='k', cmap=plt.cm.RdBu)
plt.xlabel('Standardized Length (cm)')
plt.ylabel('Standardized Yellow Score')
plt.title('Decision Boundary')
plt.legend(*scatter.legend_elements(), title="Classes")
plt.savefig('decision_boundary.png')
print("Decision boundary plot saved as 'decision_boundary.png'")

print("\nExecution completed successfully!") 