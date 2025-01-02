import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torchvision.transforms as transforms
import flwr as fl
from opacus import PrivacyEngine
import os
from grpc import RpcError
import matplotlib.pyplot as plt
import sys
import random
from PIL import Image
from collections import Counter

# Get client ID and total number of clients from command-line arguments
client_id = int(sys.argv[1])
num_clients = int(sys.argv[2])  # Pass the total number of clients as an argument
print(f"Starting Client {client_id} out of {num_clients} clients")

# Check if MPS (Apple's GPU) is available, otherwise fallback to CPU
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Function to load images from IDX files
def load_images(file_path):
    with open(file_path, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, cols, 1)
        return images

# Function to load labels from IDX files
def load_labels(file_path):
    with open(file_path, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
        return labels


# Integrate into transforms
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
#    transforms.RandomCrop(28, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


# Define custom dataset class
class MNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Convert NumPy array to PIL Image
        image = Image.fromarray(image.squeeze(), mode='L')

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)
        return image, label
        
# Base directory where the script is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Dataset path relative to the script
dataset_path = os.path.join(base_dir, "../dataset/MNIST/")
print(dataset_path)

# Load raw images and labels from the IDX files
train_images = load_images(os.path.join(dataset_path, 'train-images.idx3-ubyte'))
train_labels = load_labels(os.path.join(dataset_path, 'train-labels.idx1-ubyte'))


# Function to get Non-IID data with dynamic overlap for each round
def get_dynamic_non_iid_data(client_id, num_clients, overlap_ratio=0.2):
    num_classes = 10
    classes_per_client = num_classes // num_clients
    assigned_classes = list(range(client_id * classes_per_client, (client_id + 1) * classes_per_client))

    # Filter for assigned classes with partial overlap
    client_indices = [i for i, label in enumerate(train_labels) if label in assigned_classes]
    overlap_indices = np.random.choice(client_indices, int(overlap_ratio * len(client_indices)), replace=False)
    
    # Combine primary and overlapping samples
    client_train_images = np.concatenate((train_images[client_indices], train_images[overlap_indices]))
    client_train_labels = np.concatenate((train_labels[client_indices], train_labels[overlap_indices]))

    return client_train_images, client_train_labels


# Update the client data loading to use non-IID data
client_train_images, client_train_labels = get_dynamic_non_iid_data(client_id, num_clients, overlap_ratio=0.2)
    
# print class distributions
def print_class_distribution(labels, client_id, round_num):
    class_counts = Counter(labels)
    class_distribution_str = f"Client {client_id}, Round {round_num} - Class Distribution: {dict(class_counts)}"
    print(class_distribution_str)  # Keep the existing print functionality
    return class_distribution_str  # Return the string for logging

# Reload dynamic data with overlap for the new round
#client_train_images, client_train_labels = get_dynamic_non_iid_data(client_id, num_clients, overlap_ratio=0.3)
#client_train_images, client_train_labels = get_dynamic_non_iid_data(client_id, num_clients, overlap_ratio=0.3)
if len(client_train_images) == 0 or len(client_train_labels) == 0:
    print(f"Client {client_id} has no data to train.")



# Create client-specific dataset and DataLoader
train_dataset = MNISTDataset(client_train_images, client_train_labels, transform=transform)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

batch_size = 16  # Updated batch size

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
#val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)



# Define the CNN model
#class DeeperCNN(nn.Module):
#    def __init__(self):
#        super(DeeperCNN, self).__init__()
#        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
#        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#        self.dropout1 = nn.Dropout(0.2)
#        self.dropout2 = nn.Dropout(0.4)
#        self.fc1 = nn.Linear(128 * 7 * 7, 256)
#        self.fc2 = nn.Linear(256, 128)
#        self.fc3 = nn.Linear(128, 10)
#
#    def forward(self, x):
#        x = torch.relu(self.conv1(x))
#        x = torch.max_pool2d(torch.relu(self.conv2(x)), 2)
#        x = self.dropout1(torch.max_pool2d(torch.relu(self.conv3(x)), 2))
#        x = x.view(x.size(0), -1)
#        x = torch.relu(self.fc1(x))
#        x = self.dropout2(torch.relu(self.fc2(x)))
#        x = self.fc3(x)
#        return torch.log_softmax(x, dim=1)

## Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(4, 32)  # Replace BatchNorm with GroupNorm
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(8, 64)  # Replace BatchNorm with GroupNorm
        self.fc1_input_size = self._get_flattened_size()
        self.fc1 = nn.Linear(self.fc1_input_size, 128)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.gn1(self.conv1(x)))  # Apply GroupNorm after Conv
        x = F.max_pool2d(F.relu(self.gn2(self.conv2(x))), 2)  # Apply GroupNorm
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def _get_flattened_size(self):
        """Run a dummy input through the network to find the flattened size."""
        with torch.no_grad():
            x = torch.zeros((1, 1, 28, 28))  # Assuming MNIST input size
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            return x.numel()  # Return the total number of elements


### Define the CNN model
#class SimpleCNN(nn.Module):
#    def __init__(self):
#        super(SimpleCNN, self).__init__()
#        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
#        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#        self.fc1_input_size = self._get_flattened_size()
#        self.fc1 = nn.Linear(self.fc1_input_size, 128)
#        self.dropout = nn.Dropout(0.4)
#        self.fc2 = nn.Linear(128, 10)
#
#    def forward(self, x):
#        x = F.relu(self.conv1(x))  # Apply ReLU after Conv1
#        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # Apply ReLU after Conv2 with MaxPool
#        x = x.view(x.size(0), -1)  # Flatten
#        x = self.dropout(F.relu(self.fc1(x)))
#        x = self.fc2(x)
#        return F.log_softmax(x, dim=1)
#
#    def _get_flattened_size(self):
#        """Run a dummy input through the network to find the flattened size."""
#        with torch.no_grad():
#            x = torch.zeros((1, 1, 28, 28))  # Assuming MNIST input size
#            x = F.relu(self.conv1(x))
#            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
#            return x.numel()  # Return the total number of elements


# Initialize the CNN model
model = SimpleCNN().to(device)
#model = DeeperCNN().to(device)
print("Model initialized.")

# Directory paths for saving model
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "client_models/")
os.makedirs(base_dir, exist_ok=True)  # Ensure the directory exists
model_file_path = os.path.join(base_dir, f'simple_cnn_mnist_client_{client_id}.pth')

# Function to save the model state
def save_model(model_file_path):
    print(f"Saving model to {model_file_path}")
    torch.save(model.state_dict(), model_file_path)

# Load saved model if it exists
if os.path.exists(model_file_path):
    print(f"Model found at {model_file_path}. Loading the model weights.")
#    state_dict = torch.load(model_file_path, map_location=device)
    state_dict = torch.load(model_file_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
else:
    print(f"No pre-trained model found. Starting training from scratch.")

# Flower client class for federated learning
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.model.train()
        
        # Initialize PrivacyEngine for differential privacy
        print("Initializing PrivacyEngine...")
        self.privacy_engine = PrivacyEngine()
        self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            noise_multiplier=0.8,
            max_grad_norm=1.0,
            batch_first=True,
        )
        print("PrivacyEngine initialized successfully.")
        
        # Initialize current round counter and metrics storage
        self.current_round = 0
        self.train_losses_per_round = []
        self.train_accuracies_per_round = []
        self.val_losses_per_round = []
        self.val_accuracies_per_round = []
        self.epsilon_per_round = []  # Track epsilon per round

    def prox_term(self, global_params):
        prox_loss = 0.0
        for param, global_param in zip(self.model.parameters(), global_params):
            prox_loss += ((param - global_param.to(param.device)).norm(2))**2
        return 0.0001 * prox_loss  # FedProx parameter

    def get_parameters(self, config):
        print("Fetching model parameters...")
        return [val.cpu().detach().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters):
        print("Setting model parameters...")
        for model_param, param in zip(self.model.parameters(), parameters):
            model_param.data = torch.tensor(param, device=model_param.device)

    def train(self, epochs, global_params):
        self.model.train()

        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            print(f"Client {client_id}, Epoch {epoch+1} started.")
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(device), target.to(device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                prox_loss = self.prox_term(global_params)
                total_loss = loss + prox_loss
                total_loss.backward()
                self.optimizer.step()

                running_loss += total_loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                # Print details for each batch
                print(f"Client {client_id}, Epoch {epoch+1}, Batch {batch_idx+1}/{len(self.train_loader)}, Loss: {total_loss.item():.4f}")

            epoch_loss = running_loss / len(self.train_loader)
            epoch_accuracy = 100 * correct / total if total > 0 else 0
            self.train_losses_per_round.append(epoch_loss)
            self.train_accuracies_per_round.append(epoch_accuracy)
            print(f"Client {client_id}, Epoch {epoch+1} Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.2f}%")

            # Validation
            val_loss, val_accuracy = self.validate()
            self.val_losses_per_round.append(val_loss)
            self.val_accuracies_per_round.append(val_accuracy)
            print(f"Client {client_id}, Epoch {epoch+1} Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

#        # Visualization of training and validation metrics
#        plt.figure(figsize=(10, 5))
#        plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
#        plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
#        plt.title(f"Client {client_id} Training and Validation Loss")
#        plt.xlabel("Epochs")
#        plt.ylabel("Loss")
#        plt.legend()
#        plt.show()
#
#        plt.figure(figsize=(10, 5))
#        plt.plot(range(1, epochs + 1), train_accuracies, label='Training Accuracy')
#        plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
#        plt.title(f"Client {client_id} Training and Validation Accuracy")
#        plt.xlabel("Epochs")
#        plt.ylabel("Accuracy (%)")
#        plt.legend()
#        plt.show()

    def validate(self):
        # Ensure the model is in evaluation mode
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        print("Starting validation...")
        with torch.no_grad():  # Disable gradient computation
            for data, target in self.val_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                loss = self.criterion(output, target)
                val_loss += loss.item() * data.size(0)
                _, predicted = torch.max(output, 1)
                correct += (predicted == target).sum().item()
                total += target.size(0)

        # Calculate the average loss and accuracy
        val_loss /= total
        val_accuracy = 100 * correct / total if total > 0 else 0
        print("Validation completed.")
        return val_loss, val_accuracy




    def fit(self, parameters, config):
        # Reload dynamic data with overlap for the new round
        client_train_images, client_train_labels = get_dynamic_non_iid_data(client_id, num_clients, overlap_ratio=0.2)

        # Check if data was loaded correctly
        if len(client_train_images) == 0 or len(client_train_labels) == 0:
            print(f"Client {client_id} has no data to train for round {self.current_round + 1}.")
            return None, 0, {}

        # Update the training dataset and data loader with the new data
        train_dataset = MNISTDataset(client_train_images, client_train_labels, transform=transform)
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

        # Reinitialize the data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        # Log the class distribution and retrieve it as a string
        class_distribution_log = print_class_distribution(client_train_labels, client_id, self.current_round + 1)
        
        # Track class distributions per round
        if not hasattr(self, "class_distributions_per_round"):
            self.class_distributions_per_round = []
        self.class_distributions_per_round.append(class_distribution_log)

        # Update the round number from config if provided
        if 'round' in config:
            self.current_round = config['round']
        else:
            self.current_round += 1  # Increment if not provided
        print(f"Client {client_id} received fit request. Starting Round {self.current_round}")

        # Set model parameters and start training
        global_params = [torch.tensor(param) for param in parameters]
        self.set_parameters(parameters)
        self.train(epochs=1, global_params=global_params)
        
        # Retrieve validation loss and accuracy
        val_loss, val_accuracy = self.validate()

        # Save the model
        save_model(model_file_path)
        print(f"Model saved to {model_file_path}")

        # Retrieve privacy budget (ε) if using differential privacy
        epsilon = self.privacy_engine.get_epsilon(delta=1e-5)
        self.epsilon_per_round.append(epsilon)
        print(f"Privacy budget for Client {client_id}: ε = {epsilon:.2f}, δ = 1e-5")

        # Print per-round metrics (to console only)
        log = (
            f"Client {client_id}, Round {self.current_round}: "
            f"Class Distribution: {class_distribution_log}\n"
            f"Training Loss = {self.train_losses_per_round[-1]:.4f}, Training Accuracy = {self.train_accuracies_per_round[-1]:.2f}%\n"
            f"Validation Loss = {self.val_losses_per_round[-1]:.4f}, Validation Accuracy = {self.val_accuracies_per_round[-1]:.2f}%\n"
            f"Privacy Budget (ε) = {epsilon:.2f}, δ = 1e-5\n\n"
        )
        print(log)

        # Save only the summary metrics after the last round
        if self.current_round == config.get("num_rounds", self.current_round):
            print(f"Saving final summary for Client {client_id} after the last round...")
            with open(f"metrics evaluation/client_{client_id}_summary.txt", "w") as f:  # Overwrite mode for final summary
                for i in range(len(self.train_losses_per_round)):
                    summary_log = (
                        f"Client {client_id}, Round {i + 1}: "
                        f"Class Distribution: {self.class_distributions_per_round[i]}\n"
                        f"Training Loss = {self.train_losses_per_round[i]:.4f}, Training Accuracy = {self.train_accuracies_per_round[i]:.2f}%\n"
                        f"Validation Loss = {self.val_losses_per_round[i]:.4f}, Validation Accuracy = {self.val_accuracies_per_round[i]:.2f}%\n"
                        f"Privacy Budget (ε) = {self.epsilon_per_round[i]:.2f}, δ = 1e-5\n\n"
                    )
                    print(summary_log)
                    f.write(summary_log)
                
        return self.get_parameters(config), len(self.train_loader.dataset), {"epsilon": epsilon}



# Start the Flower client
if __name__ == "__main__":
    client = FlowerClient(model, train_loader, val_loader)
    try:
        fl.client.start_client(server_address="127.0.0.1:8080", client=client.to_client())
    except RpcError as e:
        print("Server is not reachable. Shutting down the client gracefully.")
        sys.exit(0)  # Exit the client gracefully
