import numpy as np
import matplotlib.pyplot as plt
from .tensor_basics import Tensor

class Dataset:
    """
    A simple dataset interface, similar to PyTorch's Dataset.
    This implementation demonstrates the key concepts behind PyTorch's data loading.
    """
    def __init__(self):
        pass
    
    def __getitem__(self, index):
        """Return the item at the given index."""
        raise NotImplementedError
    
    def __len__(self):
        """Return the number of items in the dataset."""
        raise NotImplementedError


class DataLoader:
    """
    A simple data loader class, similar to PyTorch's DataLoader.
    This provides batched data loading functionality.
    """
    def __init__(self, dataset, batch_size=1, shuffle=False):
        """
        Initialize the data loader.
        
        Args:
            dataset: The dataset to load data from
            batch_size: Number of samples in each batch
            shuffle: Whether to shuffle the data
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __iter__(self):
        """Return an iterator over the dataset."""
        indices = list(range(len(self.dataset)))
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            
            # Gather data from the dataset
            batch_data = [self.dataset[j] for j in batch_indices]
            
            # Separate inputs and targets if they are tuples
            if isinstance(batch_data[0], tuple):
                inputs = [item[0] for item in batch_data]
                targets = [item[1] for item in batch_data]
                
                # Convert to tensors if they aren't already
                if not isinstance(inputs[0], Tensor):
                    inputs = [Tensor(x) for x in inputs]
                if not isinstance(targets[0], Tensor):
                    targets = [Tensor(y) for y in targets]
                
                yield inputs, targets
            else:
                # Just return the data itself
                yield batch_data
    
    def __len__(self):
        """Return the number of batches."""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class SimpleDataset(Dataset):
    """
    A simple dataset of random points for demonstration.
    """
    def __init__(self, num_samples=1000):
        super().__init__()
        self.data = np.random.randn(num_samples, 2)  # 2D points (x, y)
        self.targets = np.sin(self.data[:, 0]) + 0.1 * np.random.randn(num_samples)  # y = sin(x) + noise
    
    def __getitem__(self, index):
        return self.data[index], self.targets[index]
    
    def __len__(self):
        return len(self.data)


def random_split(dataset, lengths):
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths.
    
    Args:
        dataset: Dataset to be split
        lengths: Lengths of the new datasets
        
    Returns:
        List of datasets
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of lengths must equal the length of the dataset")
    
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    
    result = []
    offset = 0
    for length in lengths:
        subset_indices = indices[offset:offset + length]
        subset = SubsetDataset(dataset, subset_indices)
        result.append(subset)
        offset += length
    
    return result


class SubsetDataset(Dataset):
    """
    Subset of a dataset at specified indices.
    """
    def __init__(self, dataset, indices):
        super().__init__()
        self.dataset = dataset
        self.indices = indices
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
    
    def __len__(self):
        return len(self.indices)


def visualize_dataset(dataset, num_samples=100):
    """Visualize a dataset."""
    plt.figure(figsize=(10, 6))
    
    # Take a random sample if dataset is large
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for idx in indices:
        inputs, targets = dataset[idx]
        plt.scatter(inputs[0], targets, alpha=0.5)
    
    plt.title('Dataset Visualization')
    plt.xlabel('x')
    plt.ylabel('y = sin(x) + noise')
    plt.grid(True, alpha=0.3)
    plt.savefig('dataset_visualization.png')
    plt.close()


def demo_dataset_loading():
    """Demonstrate dataset loading."""
    print("\n===== Dataset and Data Loading =====")
    
    # Create a dataset
    dataset = SimpleDataset(num_samples=1000)
    print(f"Dataset size: {len(dataset)}")
    
    # Get a single item
    inputs, target = dataset[0]
    print(f"Sample input: {inputs}, target: {target}")
    
    # Create a data loader
    batch_size = 32
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Number of batches: {len(data_loader)}")
    
    # Iterate over one batch
    batch_inputs, batch_targets = next(iter(data_loader))
    print(f"Batch size: {len(batch_inputs)}")
    print(f"First input in batch: {batch_inputs[0]}")
    print(f"First target in batch: {batch_targets[0]}")
    
    # Split dataset into train and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    
    # Visualize the dataset
    visualize_dataset(dataset, num_samples=200)


if __name__ == "__main__":
    demo_dataset_loading()
