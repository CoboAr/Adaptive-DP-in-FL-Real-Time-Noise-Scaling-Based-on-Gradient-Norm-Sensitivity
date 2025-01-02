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
from opacus.accountants import RDPAccountant
from opacus import PrivacyEngine, GradSampleModule
import math
from scipy.optimize import minimize
import pandas as pd
from collections import deque

# Get client ID and total number of clients from command-line arguments
client_id = int(sys.argv[1])
num_clients = int(sys.argv[2])  # Pass the total number of clients as an argument
print(f"Starting Client {client_id} out of {num_clients} clients")

# Check if MPS (Apple's GPU) is available, otherwise fallback to CPU
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Define directory for gradient analysis
base_dir = os.path.dirname(os.path.abspath(__file__))  # Script's base directory
gradient_analysis_dir = os.path.join(base_dir, "gradient_analysis/")
os.makedirs(gradient_analysis_dir, exist_ok=True)


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
    
# Print class distributions
def print_class_distribution(labels, client_id, round_num):
    class_counts = Counter(labels)
    class_distribution_str = f"Client {client_id}, Round {round_num} - Class Distribution: {dict(class_counts)}"
    print(class_distribution_str)  # Keep the existing print functionality
    return class_distribution_str  # Return the string for logging

# Check if data was loaded correctly
if len(client_train_images) == 0 or len(client_train_labels) == 0:
    print(f"Client {client_id} has no data to train.")

# Create client-specific dataset and DataLoader
train_dataset = MNISTDataset(client_train_images, client_train_labels, transform=transform)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

batch_size = 16  # Updated batch size

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)
#val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    drop_last=False
)

# Define the CNN model
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

# Initialize the CNN model
model = SimpleCNN().to(device)
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
    state_dict = torch.load(model_file_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
else:
    print(f"No pre-trained model found. Starting training from scratch.")

# Define the folder where gradient norms will be saved
gradient_norms_dir = "gradient_norms"  # New folder for gradient norms
os.makedirs(gradient_norms_dir, exist_ok=True)

# Flower client class for federated learning
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader):
        """
        Initialize the Flower client with model, data loaders, optimizer, and privacy parameters.
        Set up the variables required for training, privacy accounting, and dynamic sensitivity adjustment.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005, weight_decay=1e-4)
        self.model.train()

        # Initialize current round counter and metrics storage
        self.current_round = 0
        self.train_losses_per_round = []
        self.train_accuracies_per_round = []
        self.val_losses_per_round = []
        self.val_accuracies_per_round = []
        self.epsilon_per_round = []  # Track epsilon per round
        self.class_distributions_per_round = []

        # Define static privacy budget per round
        self.static_epsilon_per_round = [
            0.20, 0.27, 0.32, 0.36, 0.40, 0.43, 0.47, 0.50,
            0.53, 0.55, 0.58, 0.61, 0.63, 0.65, 0.68, 0.70
        ]

        # Privacy parameters
        self.max_grad_norm = 10.0  # Clipping threshold (adjust as needed)
        self.delta = 1e-5  # Privacy parameter delta
        self.sample_rate = self.train_loader.batch_size / len(self.train_loader.dataset)
        self.steps = 0  # Total number of steps (batches)
        # Orders (α) for RDP accountant
        self.orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        self.rdp = np.zeros_like(self.orders, dtype=float)  # Initialize RDP values

        # Sensitivity scale and noise multiplier parameters
        self.base_noise_multiplier = 0.3  # Initial base noise multiplier
        self.noise_factor = 0.05  # Noise factor adjustment
        self.min_noise_multiplier = 0.05  # Minimum noise multiplier
        self.max_noise_multiplier = 0.8  # Maximum noise multiplier

        # Initialize deque for sliding window of gradient norms
        self.window_size = 100  # Adjust window size as needed
        self.batch_gradients = deque(maxlen=self.window_size)

        # Initialize lists to store gradient norms and noise multipliers for logging
        self.gradient_norms_per_batch = []
        self.noise_multipliers_per_batch = []
        self.sensitivity_scales_per_batch = []

    def prox_term(self, global_params):
        """
        Compute the proximal term for FedProx optimization.
        This term penalizes the deviation from the global model parameters.

        Parameters:
            global_params (list of torch.Tensor): Global model parameters.

        Returns:
            prox_loss (torch.Tensor): The proximal term loss.
        """
        prox_loss = 0.0
        for param, global_param in zip(self.model.parameters(), global_params):
            prox_loss += ((param - global_param.to(param.device)).norm(2)) ** 2
        return 0.0001 * prox_loss  # FedProx parameter (μ)

    def get_parameters(self, config):
        """
        Retrieve the model parameters to send to the server.

        Parameters:
            config (dict): Configuration dictionary (unused).

        Returns:
            parameters (list): Model parameters as a list of NumPy arrays.
        """
        print("Fetching model parameters...")
        return [val.cpu().detach().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters):
        """
        Set the model parameters received from the server.

        Parameters:
            parameters (list): Model parameters as a list of NumPy arrays.
        """
        print("Setting model parameters...")
        for model_param, param in zip(self.model.parameters(), parameters):
            model_param.data = torch.tensor(param, device=model_param.device)

    def solve_noise_multiplier(self, n_high, n_low, n_mid, epsilon_round):
        """
        Solve for sigma_base and sigma_factor given the counts and epsilon_round.

        The constraint is:
        n_high*(sigma_base + sigma_factor)^2 + n_low*(sigma_base - sigma_factor)^2 + n_mid*(sigma_base)^2 <= epsilon_round

        We'll minimize the sum of sigma_base and sigma_factor to reduce noise while satisfying the constraint.

        Parameters:
            n_high (int): Number of high-gradient norm batches.
            n_low (int): Number of low-gradient norm batches.
            n_mid (int): Number of mid-gradient norm batches.
            epsilon_round (float): Privacy budget for the current round.

        Returns:
            sigma_base (float): Calculated base noise multiplier.
            sigma_factor (float): Calculated noise factor.
        """
        def objective(x):
            # Minimize the sum of sigma_base and sigma_factor
            return x[0] + x[1]

        def constraint(x):
            sigma_base, sigma_factor = x
            return epsilon_round - (n_high * (sigma_base + sigma_factor)**2 +
                                    n_low * (sigma_base - sigma_factor)**2 +
                                    n_mid * (sigma_base)**2)

        # Initial guesses
        initial_guess = [self.base_noise_multiplier, self.noise_factor]

        # Bounds: sigma_base and sigma_factor should be positive
        bnds = ((0.01, self.max_noise_multiplier), (0.01, self.max_noise_multiplier))

        # Define constraints
        cons = {'type': 'ineq', 'fun': constraint}

        # Perform optimization
        solution = minimize(objective, initial_guess, method='SLSQP', bounds=bnds, constraints=cons)

        if solution.success:
            sigma_base, sigma_factor = solution.x
            # Update base_noise_multiplier and noise_factor with the new values
            self.base_noise_multiplier = sigma_base
            self.noise_factor = sigma_factor
            return sigma_base, sigma_factor
        else:
            print("Optimization failed. Using default noise multipliers.")
            return self.base_noise_multiplier, self.noise_factor

    def train_model(self, global_params, round_num, epsilon_round):
        """
        Perform two-pass training:
        1. First Pass: Analyze gradient norms to determine thresholds and counts.
        2. Compute noise multipliers based on privacy budget and counts.
        3. Second Pass: Train with dynamic noise application.

        Parameters:
            global_params (list of torch.Tensor): Global model parameters.
            round_num (int): Current federated learning round number.
            epsilon_round (float): Privacy budget for the current round.
        """
        self.model.train()

        # First Pass: Analyze gradient norms
        print(f"Client {client_id}, Round {round_num}: Starting first pass for gradient analysis.")
        batch_grad_norms = []
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)

            per_sample_grad_norms = []
            for i in range(batch_size):
                self.optimizer.zero_grad()
                output = self.model(data[i].unsqueeze(0))
                loss = self.criterion(output, target[i].unsqueeze(0))
                prox_loss = self.prox_term(global_params)
                total_loss = loss + prox_loss
                total_loss.backward()

                # Compute gradient norm before clipping
                grad_norm = 0.0
                for param in self.model.parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.detach().clone().pow(2).sum().item()
                grad_norm = math.sqrt(grad_norm)
                per_sample_grad_norms.append(grad_norm)

            # Compute average gradient norm for the batch (before clipping)
            avg_grad_norm = np.mean(per_sample_grad_norms)
            batch_grad_norms.append(avg_grad_norm)

            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(self.train_loader):
                print(f"First Pass: Processed Batch {batch_idx + 1}/{len(self.train_loader)}, Avg Grad Norm: {avg_grad_norm:.4f}")

        # Compute thresholds
        threshold_low = np.percentile(batch_grad_norms, 40)
        threshold_high = np.percentile(batch_grad_norms, 70)
        print(f"Client {client_id}, Round {round_num}: Threshold Low = {threshold_low:.4f}, Threshold High = {threshold_high:.4f}")

        # Count n_high, n_low, n_mid
        n_high = sum(1 for g in batch_grad_norms if g > threshold_high)
        n_low = sum(1 for g in batch_grad_norms if g < threshold_low)
        n_mid = len(batch_grad_norms) - n_high - n_low
        N = len(batch_grad_norms)
        print(f"Client {client_id}, Round {round_num}: n_high = {n_high}, n_low = {n_low}, n_mid = {n_mid}, Total Batches = {N}")

        # Solve for sigma_base and sigma_factor
        sigma_base, sigma_factor = self.solve_noise_multiplier(n_high, n_low, n_mid, epsilon_round)
        print(f"Client {client_id}, Round {round_num}: Calculated σ_base = {sigma_base:.4f}, σ_factor = {sigma_factor:.4f}")

        # Second Pass: Train with dynamic noise application
        print(f"Client {client_id}, Round {round_num}: Starting second pass for training with dynamic noise.")
        running_loss = 0.0
        correct = 0
        total = 0

        # Reset DataLoader iterator
        self.train_loader.dataset.dataset.transform = transform  # Ensure transform is applied
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)

            # Determine noise_multiplier for this batch based on precomputed thresholds
            avg_grad_norm = batch_grad_norms[batch_idx]
            if avg_grad_norm > threshold_high:
                noise_multiplier = sigma_base + sigma_factor
            elif avg_grad_norm < threshold_low:
                noise_multiplier = sigma_base - sigma_factor
                noise_multiplier = max(noise_multiplier, self.min_noise_multiplier)
            else:
                noise_multiplier = sigma_base

            # Compute noise scale (sigma)
            noise_scale = noise_multiplier * self.max_grad_norm

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            prox_loss = self.prox_term(global_params)
            total_loss = loss + prox_loss
            total_loss.backward()

            # Clip gradients
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad = param.grad.clamp(-self.max_grad_norm, self.max_grad_norm)

            # Add Gaussian noise
            with torch.no_grad():
                for param in self.model.parameters():
                    if param.grad is not None:
                        noise = torch.normal(0, noise_scale, size=param.grad.shape).to(device)
                        param.grad += noise

            # Optimizer step
            self.optimizer.step()

            # Update privacy accountant
            self.steps += 1
            self._update_rdp(noise_multiplier)

            # Store gradient norms and noise multipliers for logging
            self.gradient_norms_per_batch.append(avg_grad_norm)
            self.noise_multipliers_per_batch.append(noise_multiplier)

            # Update training statistics
            running_loss += total_loss.item()
            total += batch_size

            # Compute training accuracy on the batch
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()

            # Log the noise multiplier for each batch
            print(
                f"Batch {batch_idx + 1}, Gradient Norm: {avg_grad_norm:.4f}, "
                f"Noise Multiplier: {noise_multiplier:.4f}"
            )

            # Print details for each batch
            avg_loss = running_loss / (batch_idx + 1)
            print(
                f"Client {client_id}, Round {round_num}, Batch {batch_idx + 1}/{len(self.train_loader)}, Loss: {avg_loss:.4f}"
            )

        # Calculate privacy budget (epsilon)
        epsilon = self._compute_epsilon()
        self.epsilon_per_round.append(epsilon)
        print(f"Client {client_id}, Round {round_num}: ε = {epsilon:.2f}, δ = {self.delta}")

        # Save gradient analysis to file
        gradient_file = os.path.join(
            gradient_analysis_dir, f"client_{client_id}_round_{round_num}_gradients.txt"
        )
        with open(gradient_file, "a") as f:  # Open in append mode
            f.write(f"Round {round_num}, Client {client_id}\n")
            f.write(f"Threshold Low: {threshold_low:.6f}\n")
            f.write(f"Threshold High: {threshold_high:.6f}\n")
            f.write(f"σ_base: {sigma_base:.6f}, σ_factor: {sigma_factor:.6f}\n")
            f.write(f"n_high: {n_high}, n_low: {n_low}, n_mid: {n_mid}, Total Batches: {N}\n\n")
        print(f"Saved gradient analysis for Client {client_id}, Round {round_num}.")

        # Save gradient norms to a file in the new folder
        self.save_gradient_norms(round_num)

        # Log round-level training statistics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_accuracy = 100 * correct / total if total > 0 else 0
        self.train_losses_per_round.append(epoch_loss)
        self.train_accuracies_per_round.append(epoch_accuracy)
        print(
            f"Client {client_id}, Round {round_num} Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.2f}%"
        )

        # Perform validation
        val_loss, val_accuracy = self.validate()
        self.val_losses_per_round.append(val_loss)
        self.val_accuracies_per_round.append(val_accuracy)
        print(
            f"Client {client_id}, Round {round_num} Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%"
        )

    def save_gradient_norms(self, round_num):
        """
        Save the gradient norms collected during training to a CSV file in the specified folder.

        Parameters:
            round_num (int): The current round number.
        """
        data = {
            'Batch': list(range(1, len(self.gradient_norms_per_batch) + 1)),
            'Gradient_Norm': self.gradient_norms_per_batch,
            'Noise_Multiplier': self.noise_multipliers_per_batch
        }
        df = pd.DataFrame(data)
        filename = f"client_{client_id}_round_{round_num}_gradient_norms.csv"
        filepath = os.path.join(gradient_norms_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Saved gradient norms for Client {client_id}, Round {round_num} to {filepath}.")

    def _update_rdp(self, noise_multiplier):
        """
        Update the Rényi Differential Privacy (RDP) accountant with the privacy cost of the current step.

        Parameters:
            noise_multiplier (float): The ratio of the noise standard deviation to the clipping threshold.
        """
        q = self.sample_rate  # Sampling rate
        steps = 1  # Incremental step update
        orders = np.array(self.orders)
        rdp_increment = self._compute_rdp(q, noise_multiplier, steps, orders)
        self.rdp += rdp_increment

    def _compute_epsilon(self):
        """
        Compute the privacy budget (epsilon) from the accumulated RDP values.

        Returns:
            epsilon (float): The computed privacy budget epsilon for the current delta.
        """
        orders = np.array(self.orders)
        rdp = self.rdp
        epsilons = (rdp - math.log(self.delta)) / (orders - 1)
        idx_opt = np.nanargmin(epsilons)
        return epsilons[idx_opt]

    def _compute_rdp(self, q, noise_multiplier, steps, orders):
        """
        Compute the RDP increment for the Gaussian mechanism with sampling.

        Parameters:
            q (float): Sampling rate.
            noise_multiplier (float): The ratio of the noise standard deviation to the clipping threshold.
            steps (int): Number of steps (batches).
            orders (array-like): Orders (alphas) at which to compute RDP.

        Returns:
            rdp_increment (np.ndarray): RDP increments for each order.
        """
        if noise_multiplier == 0:
            return np.inf * np.ones_like(orders)
        return steps * self._rdp_gaussian_mechanism(q, noise_multiplier, orders)

    def _rdp_gaussian_mechanism(self, q, sigma, orders):
        """
        Compute the RDP of the Gaussian mechanism with subsampled data.

        Parameters:
            q (float): Sampling rate.
            sigma (float): Noise multiplier.
            orders (array-like): Orders (alphas) at which to compute RDP.

        Returns:
            rdp (np.ndarray): RDP values for each order.
        """
        if q == 0:
            return np.zeros_like(orders)
        elif q == 1.0:
            return orders / (2 * sigma ** 2)
        else:
            return np.array([self._compute_rdp_order(q, sigma, order) for order in orders])

    def _compute_rdp_order(self, q, sigma, alpha):
        """
        Compute the RDP for a specific order (alpha) for the subsampled Gaussian mechanism.

        Parameters:
            q (float): Sampling rate.
            sigma (float): Noise multiplier.
            alpha (float): Order at which to compute RDP.

        Returns:
            rdp (float): RDP value for the given order.
        """
        if q == 0:
            return 0
        if np.isinf(alpha):
            return np.inf
        return (q ** 2) * alpha / (2 * sigma ** 2)

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

        # Check if the current round exceeds the predefined privacy budget list
        if self.current_round > len(self.static_epsilon_per_round):
            print(f"Client {client_id}: Current round {self.current_round} exceeds the predefined privacy budget list.")
            epsilon_round = self.static_epsilon_per_round[-1]  # Use the last epsilon value
        else:
            epsilon_round = self.static_epsilon_per_round[self.current_round - 1]

        print(f"Client {client_id} received fit request. Starting Round {self.current_round} with ε = {epsilon_round:.2f}")

        # Set model parameters
        global_params = [torch.tensor(param) for param in parameters]
        self.set_parameters(parameters)

        # Perform two-pass training
        self.train_model(global_params, self.current_round, epsilon_round)
        
        # Save the model
        save_model(model_file_path)
        print(f"Model saved to {model_file_path}")

        # Log the class distribution for this round
        class_distribution_log = print_class_distribution(client_train_labels, client_id, self.current_round)
        self.class_distributions_per_round.append(class_distribution_log)

        # Print per-round metrics (to console only)
        log = (
            f"Client {client_id}, Round {self.current_round}: "
            f"Class Distribution: {class_distribution_log}\n"
            f"Training Loss = {self.train_losses_per_round[-1]:.4f}, Training Accuracy = {self.train_accuracies_per_round[-1]:.2f}%\n"
            f"Validation Loss = {self.val_losses_per_round[-1]:.4f}, Validation Accuracy = {self.val_accuracies_per_round[-1]:.2f}%\n"
            f"Privacy Budget (ε) = {self.epsilon_per_round[-1]:.2f}, δ = 1e-5\n\n"
        )
        print(log)

        # Save only the summary metrics after the last round
#        if self.current_round == config.get("num_rounds", len(self.static_epsilon_per_round)):
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
            
        return self.get_parameters(config), len(self.train_loader.dataset), {"epsilon": epsilon_round}

# Start the Flower client
if __name__ == "__main__":
    client = FlowerClient(model, train_loader, val_loader)
    try:
        fl.client.start_client(server_address="127.0.0.1:8080", client=client.to_client())
    except RpcError as e:
        print("Server is not reachable. Shutting down the client gracefully.")
        sys.exit(0)  # Exit the client gracefully

