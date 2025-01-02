import flwr as fl
from typing import List, Tuple
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os
import struct
import numpy as np
from torch.utils.data import Dataset
from collections import OrderedDict
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F


# Define your model (the same model used in the client)
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


# Define a custom FedAvg strategy with server-side evaluation
class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, test_loader):
        super().__init__()
        self.test_loader = test_loader
#        self.model = DeeperCNN().to(device)
        self.model = SimpleCNN().to(device)
        
    # Save the global model to a file.
    def save_global_model(self, rnd):
        """Save the global model to a file."""
        # Dynamically construct the save directory relative to the script's location
        base_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(base_dir, "global_models")  # Save in a subdirectory "global_models"

        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Construct the full path for the global model file
        global_model_path = os.path.join(save_dir, f"global_model_round_{rnd}.pth")

        # Save the model
        torch.save(self.model.state_dict(), global_model_path)
        print(f"Global model saved at: {global_model_path}")
 
 

    def aggregate_fit(self, rnd, results, failures):
        if failures:
            print(f"Round {rnd}: Some clients failed during this round with {len(failures)} failures.")

        # Log ε reported by clients
        epsilons = [res.metrics["epsilon"] for _, res in results if "epsilon" in res.metrics]
        if epsilons:
            max_epsilon = max(epsilons)  # Take the maximum ε among clients for safety
            self.max_epsilon = max_epsilon  # Track max ε for this round
            if not hasattr(self, "total_epsilon"):
                self.total_epsilon = 0  # Initialize total ε if not already present
            self.total_epsilon += max_epsilon
        else:
            max_epsilon = 0
            self.max_epsilon = 0  # Ensure it’s reset if no ε is reported

#        print(f"Round {rnd}: Maximum reported privacy budget ε = {max_epsilon:.2f}")
#        print(f"Total Privacy budget across rounds: ε = {self.total_epsilon:.2f}")

        # Standard weight aggregation
        weights_results = [fl.common.parameters_to_ndarrays(res.parameters) for _, res in results]
        aggregated_weights = self.aggregate_weights(weights_results)
        self.set_model_weights(aggregated_weights)

        # Save the global model
        self.save_global_model(rnd)

        aggregated_parameters = fl.common.ndarrays_to_parameters(aggregated_weights)
        return aggregated_parameters, {"max_epsilon": max_epsilon, "total_epsilon": self.total_epsilon}


    def aggregate_weights(self, weights_results):
        print("Averaging client weights.")
        averaged_weights = [torch.zeros_like(torch.tensor(w)) for w in weights_results[0]]
        num_clients = len(weights_results)

        for weights in weights_results:
            for i, weight in enumerate(weights):
                averaged_weights[i] += torch.tensor(weight)

        averaged_weights = [w / num_clients for w in averaged_weights]
        print("Weight aggregation completed.")
        return averaged_weights

#    # weighted averaging of client model updates based on the validation accuracies of each client
#    def aggregate_weights(self, weights_results, validation_accuracies):
#        print("Weighted averaging client weights based on validation performance.")
#        total_accuracy = sum(validation_accuracies)
#        averaged_weights = [torch.zeros_like(torch.tensor(w)) for w in weights_results[0]]
#
#        for weights, accuracy in zip(weights_results, validation_accuracies):
#            for i, weight in enumerate(weights):
#                averaged_weights[i] += (accuracy / total_accuracy) * torch.tensor(weight)
#
#        print("Weighted aggregation completed.")
#        return averaged_weights


    def set_model_weights(self, aggregated_weights):
        print("Setting model weights for the global model.")
        params_dict = zip(self.model.state_dict().keys(), aggregated_weights)
        state_dict = OrderedDict({k: v.clone().detach() for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
        print("Model weights set successfully.")

    
    def aggregate_evaluate(self, rnd, results, failures):
        print(f"Round {rnd}: Evaluating the global model on the server's own test set.")
        global_loss, global_accuracy = self.predict_global_model()
#        self.visualize_predictions()
    
        # Retrieve privacy budget information from aggregate_fit
        max_epsilon = getattr(self, "max_epsilon", 0)
        total_epsilon = getattr(self, "total_epsilon", 0)
        
        print(f"Round {rnd}: Maximum reported privacy budget ε = {max_epsilon:.2f}")
        print(f"Round {rnd}: Total Privacy budget across rounds: ε = {total_epsilon:.2f}")

        # Save evaluation results and privacy budgets to a file
        with open("metrics evaluation/evaluation_metrics_server.txt", "a") as f:
            f.write(f"Round {rnd}: Loss = {global_loss:.4f}, Accuracy = {global_accuracy:.4f}, "
                    f"Max ε = {max_epsilon:.2f}, Total ε = {total_epsilon:.2f}\n")

        print(f"Round {rnd}: Server-side evaluation completed. Loss: {global_loss:.4f}, Accuracy: {global_accuracy:.4f}")
        return global_loss, {"accuracy": global_accuracy, "loss": global_loss}


    def predict_global_model(self):
        print("Evaluating the global model on the test set.")
        self.model.eval()
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        loss_fn = nn.CrossEntropyLoss()

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                loss = loss_fn(output, target)
                total_loss += loss.item() * target.size(0)
                _, predicted = torch.max(output, 1)
                total_correct += (predicted == target).sum().item()
                total_samples += target.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        print(f"Global Model Test Loss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")
        return avg_loss, accuracy

#    def visualize_predictions(self, num_images=5):
#        print("Visualizing predictions.")
#        self.model.eval()
#        data, target = next(iter(self.test_loader))
#        data, target = data[:num_images].to(device), target[:num_images].to(device)
#
#        with torch.no_grad():
#            output = self.model(data)
#            _, predicted = torch.max(output, 1)
#
#        for i in range(num_images):
#            image = data[i].cpu().squeeze().numpy()
#            plt.imshow(image, cmap='gray')
#            plt.title(f"Predicted: {predicted[i].item()}, Actual: {target[i].item()}")
#            plt.show()
#        print("Visualization of predictions completed.")

# Define device
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Define the correct test dataset path
base_dir = os.path.dirname(os.path.abspath(__file__))  # Base directory of the script
dataset_path = os.path.join(base_dir, "../dataset/MNIST/")  # Relative to the script location

# Load test dataset on the server for prediction after each round
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

test_images_file = os.path.join(dataset_path, 't10k-images.idx3-ubyte')
test_labels_file = os.path.join(dataset_path, 't10k-labels.idx1-ubyte')

def load_images(file_path):
    with open(file_path, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, cols, 1)
        return images

def load_labels(file_path):
    with open(file_path, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
        return labels

# Load the test images and labels
test_images = load_images(test_images_file)
test_labels = load_labels(test_labels_file)

# Create a custom dataset for the server-side test data
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

# Create the test dataset and loader
test_dataset = MNISTDataset(test_images, test_labels, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Define the custom FedAvg strategy with your desired parameters and test_loader for predictions
strategy = CustomFedAvg(test_loader=test_loader)

# Define server configuration
config = fl.server.ServerConfig(num_rounds=20)

# Start server
def main():
    # Dynamically construct the save directory relative to the script's location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_dir, "global_models")
    global_model_path = os.path.join(save_dir, 'global_model_round_last.pth')

    # Load the global model if it exists
    if os.path.exists(global_model_path):
        print(f"Loading global model from {global_model_path}")
#        state_dict = torch.load(global_model_path, map_location=device)
        state_dict = torch.load(global_model_path, map_location=device)
        strategy.model.load_state_dict(state_dict)
    else:
        print("No pre-trained global model found. Starting from scratch.")

    fl.server.start_server(server_address="127.0.0.1:8080", config=config, strategy=strategy)

if __name__ == "__main__":
    main()
