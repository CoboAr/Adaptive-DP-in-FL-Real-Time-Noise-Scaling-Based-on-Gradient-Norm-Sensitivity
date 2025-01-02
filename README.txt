
Adaptive Differential Privacy Project
=====================================

--------------------------------------------------------------
1. Project Description
--------------------------------------------------------------

This project explores the concept of Adaptive Differential Privacy using federated learning. It includes two models:
1. Static Noise Model: Applies a fixed noise level for privacy preservation.
2. Dynamic Noise Model: Implements dynamic noise adjustment based on gradient analysis.

Both models use federated learning on the MNIST dataset, distributed across multiple clients, with a focus on secure and privacy-preserving training.

--------------------------------------------------------------
2. Directory Structure
--------------------------------------------------------------

/Adaptive Differential Privacy/
├── static noise/
│   ├── client_models/          # Directory for storing client models
│   ├── global_models/          # Directory for storing global models
│   ├── metrics evaluation/     # Directory for storing evaluation metrics
│   ├── client.py               # Script for client logic
│   ├── launch_clients.py       # Script to launch multiple clients
│   ├── server.py               # Script for server logic
├── dynamic noise/
│   ├── client_models/          # Directory for storing client models
│   ├── global_models/          # Directory for storing global models
│   ├── gradient_analysis/      # Directory for storing gradient analysis data
│   ├── gradient_norms/         # Directory for storing gradient norms
│   ├── metrics evaluation/     # Directory for storing evaluation metrics
│   ├── client.py               # Script for client logic
│   ├── launch_clients.py       # Script to launch multiple clients
│   ├── server.py               # Script for server logic
│
├── dataset/
│   ├── MNIST/                  # Directory containing the MNIST dataset
│       ├── train-images.idx3-ubyte
│       ├── train-labels.idx1-ubyte
│       ├── t10k-images.idx3-ubyte
│       ├── t10k-labels.idx1-ubyte
│
├── Experiments results/        # Results derived from my experiment
│   ├── experiment1_results.txt
│   ├── experiment2_results.txt
│
├── Adaptive_Differential_Privacy_in_Federated_Learning__Real_Time_Noise_Scaling_Based_on_Gradient_Norm_Sensitivity.pdf  # Project report
├── README.txt                  # Project documentation
├── requirements.txt            # Python dependencies


--------------------------------------------------------------
3. Instructions to Run the Project
--------------------------------------------------------------

### Prerequisites
1. Python 3.11 or later.
2. Dependencies specified in `requirements.txt`.

### Setup
1. Open a Command Line Interface (CLI) and navigate to the project directory:

```
cd Adaptive Differential Privacy
```
2. Install Python dependencies:

```
pip install -r requirements.txt
```


### Dataset
Ensure the MNIST dataset is located in the following directory:
```
/Adaptive Differential Privacy/dataset/MNIST/
```
Download the data from this link: https://www.kaggle.com/datasets/hojjatk/mnist-dataset

After unzipping it, copy only:
```
train-images.idx3-ubyte, train-labels.idx1-ubyte, t10k-images.idx3-ubyte, t10k-labels.idx1-ubyte
```
to `/Adaptive Differential Privacy/dataset/MNIST/`

### Running the Static Noise Model
1. Navigate to the `static noise` directory:
```
cd static noise
```
2. Start the server:
```
python server.py
```
3. In a separate Command Line Interface window, launch clients:
```
python launch_clients.py
```

### Running the Dynamic Noise Model
1. Navigate to the `dynamic noise` directory:
```
cd dynamic noise
```
2. Start the server:
```
python server.py
```
3. In a separate Command Line Interface window, launch clients:
```
python launch_clients.py
```

--------------------------------------------------------------
4. Key Implementation Details
--------------------------------------------------------------

- Static Noise Model:
  - Uses a fixed noise multiplier throughout training.
  - Suitable for environments with predefined privacy budgets.

- Dynamic Noise Model:
  - Adjusts noise levels dynamically based on gradient analysis.
  - Enhances the trade-off between privacy and model utility.

- Server:
  - Manages the global model and aggregates weights from clients.
  - Implements a custom FedAvg strategy for model aggregation.
  - Saves global models in the respective `global_models/` directory.

- Client:
  - Processes local training using the MNIST dataset.
  - Reports training metrics and model updates to the server.

--------------------------------------------------------------
5. Results and Metrics
--------------------------------------------------------------

- Static Noise Model:
  - Server-side evaluation metrics (loss and accuracy and privacy budget) are saved in:
  ```
  /Adaptive Differential Privacy/static noise/metrics evaluation/evaluation_metrics_server.txt
  ```

- Dynamic Noise Model:
  - Server-side evaluation metrics (loss and accuracy and privacy budget) are saved in:
  ```
  /Adaptive Differential Privacy/dynamic noise/metrics evaluation/evaluation_metrics_server.txt
  ```
