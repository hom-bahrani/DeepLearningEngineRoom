import numpy as np
import matplotlib.pyplot as plt
from .tensor_basics import Tensor
from .datasets import SimpleDataset, DataLoader, random_split

class Module:
    """
    Base class for all neural network modules.
    """
    def __init__(self):
        self._parameters = {}
        self._modules = {}
    
    def forward(self, *args, **kwargs):
        """
        Forward pass computation. Should be implemented by subclasses.
        """
        raise NotImplementedError
    
    def __call__(self, *args, **kwargs):
        """
        Make the module callable.
        """
        return self.forward(*args, **kwargs)
    
    def parameters(self):
        """
        Return all parameters in the module.
        """
        params = list(self._parameters.values())
        
        for module in self._modules.values():
            params.extend(module.parameters())
        
        return params


class Linear(Module):
    """
    Linear (fully connected) layer implementation.
    Computes: y = x @ W^T + b
    """
    def __init__(self, in_features, out_features):
        """
        Initialize the linear layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
        """
        super().__init__()
        
        # Initialize weights and bias
        # Xavier/Glorot initialization for weights
        limit = np.sqrt(6 / (in_features + out_features))
        weights = np.random.uniform(-limit, limit, (out_features, in_features))
        bias = np.zeros(out_features)
        
        # Store as parameters
        self._parameters['weight'] = Tensor(weights, requires_grad=True)
        self._parameters['bias'] = Tensor(bias, requires_grad=True)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, in_features)
            
        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        # Handle case where x is a list of tensors (batch processing)
        if isinstance(x, list):
            # Stack tensors along a new batch dimension
            batch_data = np.stack([t.data for t in x])
            x = Tensor(batch_data)
        
        # Compute output: y = x @ W^T + b
        output = x.matmul(Tensor(self._parameters['weight'].data.T)) + self._parameters['bias']
        return output


class ReLU(Module):
    """
    Rectified Linear Unit (ReLU) activation function.
    """
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with ReLU activation applied
        """
        # Handle case where x is a list of tensors
        if isinstance(x, list):
            return [self.forward(t) for t in x]
        
        # ReLU: max(0, x)
        # For simplicity, let's just apply ReLU directly
        result = Tensor(np.maximum(0, x.data))
        
        # Transfer gradient information if needed
        if x.requires_grad:
            result.requires_grad = True
            mask = x.data > 0
            
            def _backward():
                if x.grad is None:
                    x.grad = np.zeros_like(x.data)
                x.grad += result.grad * mask
            
            result.grad_fn = _backward
            result._children = [(x, 'relu')]
            result._op_name = "relu"
        
        return result


class Sequential(Module):
    """
    Sequential container for modules.
    """
    def __init__(self, *modules):
        super().__init__()
        for i, module in enumerate(modules):
            self._modules[str(i)] = module
    
    def forward(self, x):
        """
        Forward pass through all modules in sequence.
        
        Args:
            x: Input tensor
            
        Returns:
            Output after passing through all modules
        """
        for module in self._modules.values():
            x = module(x)
        return x


class MSELoss:
    """
    Mean Squared Error Loss.
    """
    def __call__(self, predictions, targets):
        """
        Compute the MSE loss.
        
        Args:
            predictions: Predicted values
            targets: Target values
            
        Returns:
            MSE loss
        """
        # Handle case where inputs are lists of tensors
        if isinstance(predictions, list) and isinstance(targets, list):
            predictions_data = np.array([p.data for p in predictions])
            targets_data = np.array([t.data for t in targets])
            
            # Calculate the squared differences
            squared_diff = (predictions_data - targets_data) ** 2
            
            # Calculate mean
            loss_value = np.mean(squared_diff)
            
            # Wrap in a tensor for gradient tracking
            result = Tensor(loss_value, requires_grad=True)
            
            # Set up gradient tracking
            def _backward():
                n = len(predictions)
                for i, pred in enumerate(predictions):
                    if pred.requires_grad:
                        if pred.grad is None:
                            pred.grad = np.zeros_like(pred.data)
                        # dL/dp = 2 * (p - t) / n
                        pred.grad += 2 * (pred.data - targets_data[i]) / n
            
            result.grad_fn = _backward
            return result
        
        # Handle case where predictions is a Tensor and targets is a list of Tensors
        elif isinstance(predictions, Tensor) and isinstance(targets, list) and all(isinstance(t, Tensor) for t in targets):
            targets_data = np.array([t.data for t in targets])
            
            # If predictions is a single tensor but we need to compare with multiple targets
            # reshape it to match the targets
            if len(predictions.data.shape) == 1 and predictions.data.shape[0] == 1:
                pred_data = np.repeat(predictions.data, len(targets))
            else:
                pred_data = predictions.data
                
            # Calculate the squared differences
            squared_diff = (pred_data - targets_data) ** 2
            
            # Calculate mean
            loss_value = np.mean(squared_diff)
            
            # Wrap in a tensor for gradient tracking
            result = Tensor(loss_value, requires_grad=True)
            
            # Set up gradient tracking
            if predictions.requires_grad:
                def _backward():
                    if predictions.grad is None:
                        predictions.grad = np.zeros_like(predictions.data)
                    # Average gradient from all targets
                    predictions.grad += np.mean(2 * (pred_data - targets_data))
                
                result.grad_fn = _backward
            
            return result
        
        # If we have single tensors
        elif isinstance(predictions, Tensor) and isinstance(targets, Tensor):
            pred_data = predictions.data
            target_data = targets.data
            
            # Calculate the squared differences
            squared_diff = (pred_data - target_data) ** 2
            
            # Calculate mean
            loss_value = np.mean(squared_diff)
            
            # Wrap in a tensor for gradient tracking
            result = Tensor(loss_value, requires_grad=True)
            
            # Set up gradient tracking
            if predictions.requires_grad:
                def _backward():
                    if predictions.grad is None:
                        predictions.grad = np.zeros_like(predictions.data)
                    # dL/dp = 2 * (p - t) / n
                    predictions.grad += 2 * (pred_data - target_data) / np.prod(pred_data.shape)
                
                result.grad_fn = _backward
            
            return result
        
        # Default fallback using numpy
        raise TypeError(f"Unsupported types for MSE loss calculation: predictions={type(predictions)}, targets={type(targets)}")


class SGD:
    """
    Stochastic Gradient Descent optimizer.
    """
    def __init__(self, parameters, lr=0.01):
        """
        Initialize the SGD optimizer.
        
        Args:
            parameters: Parameters to optimize
            lr: Learning rate
        """
        self.parameters = parameters
        self.lr = lr
    
    def zero_grad(self):
        """Zero out the gradients of all parameters."""
        for param in self.parameters:
            if param.grad is not None:
                param.grad = None
    
    def step(self):
        """Update parameters based on gradients."""
        for param in self.parameters:
            if param.grad is not None:
                param.data -= self.lr * param.grad


def train_one_epoch(model, dataloader, loss_fn, optimizer):
    """Train the model for one epoch."""
    epoch_loss = 0.0
    num_batches = 0
    
    for inputs, targets in dataloader:
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Compute loss
        loss = loss_fn(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Track loss
        epoch_loss += loss.data.item()
        num_batches += 1
    
    return epoch_loss / num_batches


def demo_neural_network():
    """Demonstrate a simple neural network."""
    print("\n===== Neural Network Training =====")
    
    # Create a dataset
    dataset = SimpleDataset(num_samples=1000)
    
    # Split into train and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create a simple model
    model = Sequential(
        Linear(2, 16),
        ReLU(),
        Linear(16, 8),
        ReLU(),
        Linear(8, 1)
    )
    
    # Print model structure
    print("Model structure:")
    for name, module in model._modules.items():
        print(f"  Layer {name}: {module.__class__.__name__}")
        if isinstance(module, Linear):
            weight_shape = module._parameters['weight'].shape
            print(f"    Weight shape: {weight_shape}")
            print(f"    Bias shape: {module._parameters['bias'].shape}")
    
    # Define loss function and optimizer
    loss_fn = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.01)
    
    # Train the model
    print("\nTraining the model...")
    epochs = 5
    train_losses = []
    
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
        train_losses.append(train_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}")
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), train_losses, 'o-')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig('training_loss.png')
    plt.close()
    
    # Evaluate the model
    print("\nEvaluating the model...")
    test_loss = 0.0
    num_batches = 0
    
    for inputs, targets in test_loader:
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        test_loss += loss.data.item()
        num_batches += 1
    
    test_loss /= num_batches
    print(f"Test Loss: {test_loss:.4f}")
    
    # Visualize predictions
    plt.figure(figsize=(10, 6))
    
    # Get sample predictions
    inputs, targets = next(iter(test_loader))
    outputs = model(inputs)
    
    # Convert to numpy for plotting
    inputs_np = np.array([inp.data for inp in inputs])
    targets_np = np.array([t.data for t in targets])
    
    # Handle case where outputs is a single Tensor
    if isinstance(outputs, Tensor):
        outputs_np = outputs.data.reshape(-1)  # Reshape to match targets if needed
    else:
        outputs_np = np.array([out.data for out in outputs])
    
    # Plot
    plt.scatter(inputs_np[:, 0], targets_np, label='True values', alpha=0.5)
    plt.scatter(inputs_np[:, 0], outputs_np, label='Predictions', alpha=0.5)
    plt.title('Model Predictions vs. True Values')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('model_predictions.png')
    plt.close()


if __name__ == "__main__":
    demo_neural_network()
