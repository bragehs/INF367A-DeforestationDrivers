import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

class SatelliteSegmentation:
    def __init__(self, train_data, train_labels, test_split=0.2, batch_size=1, learning_rate=0.001):
        """
        Initialize the SatelliteSegmentation class.
        
        Args:
            train_data (np.ndarray): Training data of shape (num_images, num_bands, height, width).
            train_labels (np.ndarray): Training labels of shape (num_images, height, width).
            test_split (float): Fraction of data to use for testing.
            batch_size (int): Batch size for DataLoader.
            learning_rate (float): Learning rate for the optimizer.
        """
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_split = test_split
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Initialize dataset and data loaders
        self._prepare_data()
        
        # Initialize model, loss, and optimizer
        self.model = self._build_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def _prepare_data(self):
        """Prepare the dataset and split it into training and testing sets."""
        dataset = self.SatelliteDataset(self.train_data, self.train_labels)
        
        # Split dataset into train and test
        train_size = int((1 - self.test_split) * len(dataset))
        test_size = len(dataset) - train_size
        self.train_dataset, self.test_dataset = random_split(dataset, [train_size, test_size])
        
        # Create DataLoaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
    
    def _build_model(self):
        """Build a simple segmentation model."""
        class SimpleSegmentationModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(12, 5, kernel_size=1)  # 12 bands to 5 classes
            
            def forward(self, x):
                return self.conv(x)
        
        return SimpleSegmentationModel()
    
    def train(self, num_epochs=10):
        """Train the model."""
        self.model.train()
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            
            for images, labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(self.train_loader):.4f}")
    
    def evaluate(self):
        """Evaluate the model on the test set."""
        self.model.eval()
        total_correct = 0
        total_pixels = 0
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                outputs = self.model(images)
                _, predictions = torch.max(outputs, 1)
                total_correct += (predictions == labels).sum().item()
                total_pixels += labels.numel()
        
        accuracy = 100.0 * total_correct / total_pixels
        print(f"Test Accuracy: {accuracy:.2f}%")
        return accuracy
    
    class SatelliteDataset(Dataset):
        """Custom Dataset class for satellite images and labels."""
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            image = torch.tensor(self.data[idx], dtype=torch.float32)
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return image, label
