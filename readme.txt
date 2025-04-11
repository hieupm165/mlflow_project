# MLflow Classification Project

This project demonstrates a machine learning classification pipeline using MLflow for experiment tracking, model management, and deployment.

## Project Structure

```
mlflow_project/
├── MLproject        # MLflow project definition
├── conda.yaml       # Conda environment file
├── train.py         # Training script
├── app.py           # Flask web application
├── Dockerfile       # Docker configuration
├── requirements.txt # Python dependencies
└── .gitlab-ci.yml   # GitLab CI/CD configuration
```

## Getting Started

### Prerequisites

- Python 3.9+
- Conda or virtualenv
- Docker (for containerization)
- GitLab account (for CI/CD)

### Installation

1. Clone the repository:
   ```bash
   git clone https://gitlab.com/your-username/MSE_DDM501.git
   cd MSE_DDM501
   ```

2. Create and activate a conda environment:
   ```bash
   conda env create -f conda.yaml
   conda activate classification-env
   ```

3. Run the training script to generate models:
   ```bash
   python train.py
   ```

4. Start the Flask web application:
   ```bash
   python app.py
   ```

5. Access the web application at http://localhost:5000

## MLflow Experiment Tracking

All experiments are tracked using MLflow. You can view the experiments by starting the MLflow UI:

```bash
mlflow ui
```

Then access the MLflow UI at http://localhost:5000 to view experiment results.

## Docker Deployment

Build and run the Docker container locally:

```bash
docker build -t mse_ddm501:latest .
docker run -p 5000:5000 mse_ddm501:latest
```

## GitLab CI/CD Setup

1. Create a new repository on GitLab named "MSE_DDM501"
2. Add the following environment variables in GitLab CI/CD settings:
   - DOCKER_USERNAME: Your Docker Hub username
   - DOCKER_TOKEN: Your Docker Hub access token

3. Push your code to the GitLab repository:
   ```bash
   git push origin main
   ```

4. The CI/CD pipeline will automatically build and push the Docker image to Docker Hub when you push to the main branch.

## Running the Deployed Application

Once the Docker image is pushed to Docker Hub, you can run it on any machine with Docker:

```bash
docker run -p 5000:5000 your-dockerhub-username/mse_ddm501:latest
```

Then access the application at http://localhost:5000

## License

This project is licensed under the MIT License - see the LICENSE file for details.
