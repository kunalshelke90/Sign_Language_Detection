﻿# Sign Language Detection with YOLOv5

Welcome to the Sign Language Detection Project! This is the sign language detection project using YOLOv5, Jenkins,Flask, and AWS for deployment. Dive into the world of computer vision and machine learning to detect and interpret sign language gestures.

## Project Architecture

Our project is structured into the following key components:

- **Constants**: Houses the static variables utilized across the project.
- **Config Entity**: Contains configuration settings for various components.
- **Artifact Entity**: Manages the storage and lifecycle of model artifacts.
- **Components**: Consists of modules handling different stages of the detection pipeline.
- **Pipeline**: Manages the end-to-end flow from data ingestion to model prediction.
- **app.py**: The main application file for interacting with the detection model.

## Prerequisites

Before getting started, ensure you have installed the following tools:

- **AWS CLI**: Essential for model deployment using S3.
- **Jenkins**: For continuous integration and automated deployment tasks.
- **YOLOv5**: The model framework used for sign language detection.
- **Flask**: Used for it frontend page.

### Installing AWS CLI

Visit the [AWS CLI installation guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) to install the AWS CLI.

### Configuring AWS Credentials

Configure your AWS credentials with your secret key and access key. Run the following command in your terminal:

```bash
aws configure
```

Follow the prompts to input your AWS Access Key ID, AWS Secret Access Key, and default region name.

### Setting Up an S3 Bucket

Create an S3 bucket for the model deployment. The bucket name should correspond with the name mentioned in the project's constants.

## Jenkins Setup

Configure Jenkins for continuous integration:

1. **Jenkins**: create EC2 on over there setup jenkis

2. **Create a New Job**: Setup a new project/job in Jenkins for this repository.

3. **Integrate with Repository**: Connect Jenkins with your version control system (GitHub, GitLab, etc.).

4. **Build and Deploy**: Configure Jenkins to build the project and deploy the model to AWS on every commit.

## How to Run the Project

Follow these steps to set up and run the project environment:

1. **Create a Conda Environment**

   Create a new Conda environment with Python 3.8:

   ```bash
   conda create -n venv python=3.8 -y
   ```

2. **Activate the Environment**

   Activate your Conda environment:

   ```bash
   conda activate venv
   ```

3. **Install Dependencies**

   Install all required dependencies from the `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**

   Initiate the `app.py` file to start the detection application:

   ```bash
   python app.py
   ```

## Additional Features

- **Cloud Deployment**: Leverage AWS for deploying your model for broader access.
- **Continuous Integration**: Automate testing and deployments with Jenkins.
- **YOLOv5**: Utilize the power of YOLOv5 for fast and accurate sign language gesture detection.



