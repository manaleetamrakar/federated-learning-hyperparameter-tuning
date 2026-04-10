# Intelligent Hyperparameter Optimizer

Intelligent Hyperparameter Optimizer is a full-stack machine learning application designed to dynamically adjust training parameters based on system hardware, ensuring efficient, stable, and crash-free model training.

---

## Project Overview

This project addresses the limitations of traditional hyperparameter tuning methods that rely on static configurations. Such approaches often lead to inefficient training, unstable convergence, and system crashes due to resource constraints.

The system introduces a hardware-aware optimization approach that analyzes device specifications such as RAM, CPU, GPU, and VRAM. Based on this analysis, it dynamically configures hyperparameters like batch size, number of epochs, and optimizer selection to maximize performance while maintaining system stability.

---

## Technologies Used

- Python  
- FastAPI  
- TensorFlow / Keras  
- React (Next.js)  
- Tailwind CSS  
- Recharts  
- WebSockets  
- Docker  
- GitHub  

---

## Key Features

- Dynamically adjusts hyperparameters based on system hardware  
- Prevents out-of-memory (OOM) crashes during training  
- Streams real-time training metrics such as accuracy and loss  
- Visualizes performance using interactive graphs  
- Generates confusion matrix for detailed evaluation  
- Supports dataset selection and custom dataset upload  
- Scalable and containerized full-stack architecture  

---

## Workflow

- The user inputs system specifications such as RAM, CPU cores, GPU type, and VRAM  
- A dataset is selected or uploaded by the user  
- The backend evaluates hardware constraints and dataset characteristics  
- Hyperparameters such as batch size, epochs, and optimizer are dynamically configured  
- The model is trained using TensorFlow  
- Training metrics including accuracy and loss are streamed in real-time  
- Final results are displayed using graphs and a confusion matrix  

---

## Results

- Achieves high accuracy (typically between 90% and 96%)  
- Maintains stable convergence during training  
- Prevents crashes caused by improper hyperparameter configurations  
- Efficiently utilizes system resources across different hardware setups  

---

## Project Structure

- backend/ – FastAPI backend and machine learning logic  
- frontend/ – React-based user interface  
- docker-compose.yml – Containerized setup for full system  

---

## Setup

To run the project locally:

docker-compose up --build  

Ensure Docker is installed and running before executing the command.

---

## Status

Project completed and fully functional with real-time training visualization and hardware-aware optimization.

---

## Note

This project was initially conceptualized as part of research in federated learning and hyperparameter tuning. It was later extended into a practical system focused on hardware-aware optimization for machine learning training.
