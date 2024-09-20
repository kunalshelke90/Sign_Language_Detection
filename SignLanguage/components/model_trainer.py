# import os
# import sys
# import yaml
# import zipfile
# import shutil
# from SignLanguage.utils.main_utils import read_yaml_file
# from SignLanguage.logger import logging
# from SignLanguage.exception import SignException
# from SignLanguage.entity.config_entity import ModelTrainerConfig
# from SignLanguage.entity.artifacts_entity import ModelTrainerArtifact

# class ModelTrainer:
#     def __init__(self, model_trainer_config: ModelTrainerConfig):
#         self.model_trainer_config = model_trainer_config

#     def initiate_model_trainer(self) -> ModelTrainerArtifact:
#         logging.info("Entered initiate_model_trainer method of ModelTrainer class")

#         try:
#             logging.info("Unzipping data")

#             zip_file_path = "Sign_language_data.zip"
#             unzip_dir = ""

#             try:
#                 with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
#                     zip_ref.extractall(unzip_dir)
#                     print(f"Successfully unzipped to {unzip_dir}")
#             except zipfile.BadZipFile:
#                 raise SignException("The zip file is corrupted or not a zip file", sys)
#             except FileNotFoundError:
#                 raise SignException(f"{zip_file_path} not found", sys)

#             with open("data.yaml", 'r') as stream:
#                 num_classes = str(yaml.safe_load(stream)['nc'])

#             model_config_file_name = self.model_trainer_config.weight_name.split(".")[0]
#             print(model_config_file_name)

#             config = read_yaml_file(f"yolov5/models/{model_config_file_name}.yaml")

#             config['nc'] = int(num_classes)

#             with open(f'yolov5/models/custom_{model_config_file_name}.yaml', 'w') as f:
#                 yaml.dump(config, f)

#             # Replace os.system("cp ...") with shutil.copy
#             os.system(f"cd yolov5; python train.py --img 416 --batch {self.model_trainer_config.batch_size} --epochs {self.model_trainer_config.no_epochs} --data ../data.yaml --cfg ./models/custom_yolov5s.yaml --weights {self.model_trainer_config.weight_name} --name yolov5s_results  --cache")

#             shutil.copy("yolov5/runs/train/yolov5s_results/weights/best.pt", "yolov5/")
#             os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
#             shutil.copy("yolov5/runs/train/yolov5s_results/weights/best.pt", self.model_trainer_config.model_trainer_dir)

#             # Replace os.system("rm ...") with shutil.rmtree and os.remove
#             shutil.rmtree('yolov5/runs', ignore_errors=True)
#             shutil.rmtree('train', ignore_errors=True)
#             shutil.rmtree('test', ignore_errors=True)
#             os.remove('data.yaml')

#             model_trainer_artifact = ModelTrainerArtifact(
#                 trained_model_file_path="yolov5/best.pt",
#             )

#             logging.info("Exited initiate_model_trainer method of ModelTrainer class")
#             logging.info(f"Model trainer artifact: {model_trainer_artifact}")

#             return model_trainer_artifact

#         except Exception as e:
#             raise SignException(e, sys)
import os
import sys
import yaml
import zipfile
import shutil
import subprocess
import glob 
from SignLanguage.utils.main_utils import read_yaml_file
from SignLanguage.logger import logging
from SignLanguage.exception import SignException
from SignLanguage.entity.config_entity import ModelTrainerConfig
from SignLanguage.entity.artifacts_entity import ModelTrainerArtifact

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig):
        self.model_trainer_config = model_trainer_config

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            logging.info("Unzipping data")

            zip_file_path = "Sign_language_data.zip"
            unzip_dir = ""

            # Check if the zip file exists
            if not os.path.exists(zip_file_path):
                raise SignException(f"{zip_file_path} not found", sys)

            # Unzip the data
            try:
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    zip_ref.extractall(unzip_dir)
                    print(f"Successfully unzipped to {unzip_dir}")
            except zipfile.BadZipFile:
                raise SignException("The zip file is corrupted or not a zip file", sys)

            # Check if data.yaml exists
            data_yaml_path = os.path.join(unzip_dir, "data.yaml")
            if not os.path.exists("data.yaml"):
                raise SignException("data.yaml not found after unzipping!", sys)

            # Read number of classes from data.yaml
            with open("data.yaml", 'r') as stream:
                num_classes = str(yaml.safe_load(stream)['nc'])

            model_config_file_name = self.model_trainer_config.weight_name.split(".")[0]
            print(f"Model config file name: {model_config_file_name}")

            # Read the model YAML configuration
            model_yaml_path = f"yolov5/models/{model_config_file_name}.yaml"
            if not os.path.exists(model_yaml_path):
                raise SignException(f"Model YAML file not found: {model_yaml_path}", sys)

            config = read_yaml_file(model_yaml_path)
            config['nc'] = int(num_classes)

            # Write the modified configuration
            custom_model_yaml_path = f'yolov5/models/custom_{model_config_file_name}.yaml'
            with open(custom_model_yaml_path, 'w') as f:
                yaml.dump(config, f)

            # Prepare the training command
            train_command = [
                "python", "train.py",
                "--img", "416",
                "--batch", str(self.model_trainer_config.batch_size),
                "--epochs", str(self.model_trainer_config.no_epochs),
                "--data", "../data.yaml",
                "--cfg", f"./models/custom_{model_config_file_name}.yaml",
                "--weights", self.model_trainer_config.weight_name,
                "--name", "yolov5s_results",
                "--cache"
            ]

            # Run the training command using subprocess
            logging.info("Starting model training...")
            result = subprocess.run(
                train_command,
                cwd='yolov5',
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Check if training was successful
            if result.returncode != 0:
                logging.error("Training failed with the following error:")
                logging.error(result.stderr)
                raise SignException("Model training failed", sys)
            else:
                logging.info("Training completed successfully.")
                logging.info(result.stdout)

            # Paths to the trained model
            # trained_model_src = os.path.join("yolov5", "runs", "train", "yolov5s_results4", "weights", "best.pt")
            # Path to where YOLOv5 training results are saved
            results_dir = os.path.join("yolov5", "runs", "train")

            # Find all directories that start with 'yolov5s_results'
            result_dirs = glob.glob(os.path.join(results_dir, "yolov5s_results*"))

            # Sort directories by modification time (latest first)
            latest_result_dir = max(result_dirs, key=os.path.getmtime)

            # Path to the trained model
            trained_model_src = os.path.join(latest_result_dir, "weights", "best.pt")
            trained_model_dst = os.path.join("yolov5", "best.pt")

            # Check if the trained model exists
            if not os.path.exists(trained_model_src):
                raise SignException(f"Trained model file not found at {trained_model_src}", sys)

            # Copy the trained model to desired locations
            shutil.copy(trained_model_src, trained_model_dst)
            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
            shutil.copy(trained_model_src, self.model_trainer_config.model_trainer_dir)

            # Clean up unnecessary files and directories
            shutil.rmtree(os.path.join('yolov5', 'runs'), ignore_errors=True)
            shutil.rmtree('train', ignore_errors=True)
            shutil.rmtree('test', ignore_errors=True)
            if os.path.exists('data.yaml'):
                os.remove('data.yaml')

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=trained_model_dst,
            )

            logging.info("Exited initiate_model_trainer method of ModelTrainer class")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact

        except Exception as e:
            raise SignException(e, sys)
