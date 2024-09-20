import sys, os,shutil
from SignLanguage.pipeline.training_pipeline import TrainPipeline
from SignLanguage.exception import SignException
from SignLanguage.utils.main_utils import decodeImage, encodeImageIntoBase64
from flask import Flask, request, jsonify, render_template,Response
from flask_cors import CORS, cross_origin


app = Flask(__name__)
CORS(app)



class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"


@app.route("/")
def home():
    return render_template("index.html")



@app.route("/train")
def trainRoute():
    obj = TrainPipeline()
    obj.run_pipeline()
    return "Training Successfull!!" 




@app.route("/predict", methods=['POST','GET'])
@cross_origin()
def predictRoute():
    try:
        image = request.json['image']
        decodeImage(image, clApp.filename)

        os.system("cd yolov5/ && python detect.py --weights best.pt --img 416 --conf 0.5 --source ../data/inputImage.jpg")

        opencodedbase64 = encodeImageIntoBase64("yolov5/runs/detect/exp/inputImage.jpg")
        result = {"image": opencodedbase64.decode('utf-8')}
        # os.system("rm -rf yolov5/runs")
        shutil.rmtree('yolov5/runs', ignore_errors=True)
    except ValueError as val:
        print(val)
        return Response("Value not found inside  json data")
    except KeyError:
        return Response("Key value error incorrect key passed")
    except Exception as e:
        print(e)
        result = "Invalid input"

    return jsonify(result)




@app.route("/live", methods=['GET'])
@cross_origin()
def predictLive():
    try:
        os.system("cd yolov5/ && python detect.py --weights best.pt --img 416 --conf 0.5 --source 0")
        os.system("rm -rf yolov5/runs")
        return "Camera starting!!" 

    except ValueError as val:
        print(val)
        return Response("Value not found inside  json data")
    



if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host="0.0.0.0", port=8080)

# import sys, os
# from SignLanguage.pipeline.training_pipeline import TrainPipeline
# from SignLanguage.exception import SignException
# from SignLanguage.utils.main_utils import decodeImage, encodeImageIntoBase64
# from flask import Flask, request, jsonify, render_template, Response
# from flask_cors import CORS, cross_origin
# import shutil

# app = Flask(__name__)
# CORS(app)

# class ClientApp:
#     def __init__(self):
#         self.filename = "inputImage.jpg"

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/train")
# def trainRoute():
#     obj = TrainPipeline()
#     obj.run_pipeline()
#     return "Training Successful!!"

# @app.route("/predict", methods=['POST', 'GET'])
# @cross_origin()
# def predictRoute():
#     try:
#         image = request.json['image']
#         decodeImage(image, clApp.filename)

#         os.system("cd yolov5/ && python detect.py --weights my_model.pt --img 416 --conf 0.5 --source ../data/inputImage.jpg")

#         opencodedbase64 = encodeImageIntoBase64("yolov5/runs/detect/exp/inputImage.jpg")
#         result = {"image": opencodedbase64.decode('utf-8')}
        
#         # Remove the directory with Python
#         shutil.rmtree('yolov5/runs', ignore_errors=True)

#     except ValueError as val:
#         print(val)
#         return Response("Value not found inside JSON data")
#     except KeyError:
#         return Response("Key value error: incorrect key passed")
#     except Exception as e:
#         print(e)
#         result = "Invalid input"

#     return jsonify(result)

# @app.route("/live", methods=['GET'])
# @cross_origin()
# def predictLive():
#     try:
#         os.system("cd yolov5/ && python detect.py --weights my_model.pt --img 416 --conf 0.5 --source 0")
        
#         # Remove the directory with Python
#         shutil.rmtree('yolov5/runs', ignore_errors=True)
        
#         return "Camera starting!!"

#     except ValueError as val:
#         print(val)
#         return Response("Value not found inside JSON data")

# if __name__ == "__main__":
#     clApp = ClientApp()
#     app.run(host="0.0.0.0", port=8080)

# import sys, os ,re
# from SignLanguage.pipeline.training_pipeline import TrainPipeline
# from SignLanguage.exception import SignException
# from SignLanguage.utils.main_utils import decodeImage, encodeImageIntoBase64
# from flask import Flask, request, jsonify, render_template, Response
# from flask_cors import CORS, cross_origin
# import shutil

# app = Flask(__name__)
# CORS(app)

# class ClientApp:
#     def __init__(self):
#         self.filename = "inputImage.jpg"

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/train")
# def trainRoute():
#     obj = TrainPipeline()
#     obj.run_pipeline()
#     return "Training Successful!!"

# # @app.route("/predict", methods=['POST', 'GET'])
# # @cross_origin()
# # def predictRoute():
# #     try:
# #         image = request.json['image']
# #         decodeImage(image, clApp.filename)

# #         os.system("cd yolov5/ && python detect.py --weights best.pt --img 416 --conf 0.5 --source ../data/inputImage.jpg")
        
# #         # Fetch the last 'exp*' directory created in 'runs/detect' folder
# #         output_dir = max([d for d in os.listdir('yolov5/runs/detect') if d.startswith('exp')], key=lambda x: int(x[3:]), default=None)

# #         if output_dir is not None:
# #             output_path = os.path.join('yolov5', 'runs', 'detect', output_dir, 'inputImage.jpg')
# #             opencodedbase64 = encodeImageIntoBase64(output_path)
# #             result = {"image": opencodedbase64.decode('utf-8')}
# #         else:
# #             raise FileNotFoundError("Output not found")

# #         # Remove the runs directory
# #         shutil.rmtree('yolov5/runs', ignore_errors=True)

# #     except ValueError as val:
# #         print(val)
# #         return Response("Value not found inside JSON data")
# #     except KeyError:
# #         return Response("Key value error: incorrect key passed")
# #     except Exception as e:
# #         print(e)
# #         result = "Invalid input"

# #     return jsonify(result)

# @app.route("/predict", methods=['POST', 'GET'])
# @cross_origin()
# def predictRoute():
#     try:
#         image = request.json['image']
#         decodeImage(image, clApp.filename)

#         # Run the YOLOv5 detection
#         os.system("cd yolov5/ && python detect.py --weights best.pt --img 416 --conf 0.5 --source ../data/inputImage.jpg")

#         # List the 'exp*' directories in 'runs/detect' folder and log the output directories for debugging
#         exp_dirs = [d for d in os.listdir('yolov5/runs/detect') if d.startswith('exp')]
#         print(f"Available directories: {exp_dirs}")  # Log available directories

#         if exp_dirs:
#             # Get the latest 'exp*' directory based on numeric suffix
#             output_dir = max(exp_dirs, key=lambda x: int(x[3:]), default=None)
#             print(f"Selected directory: {output_dir}")  # Log the selected directory

#             if output_dir is not None:
#                 output_path = os.path.join('yolov5', 'runs', 'detect', output_dir, 'inputImage.jpg')
#                 if os.path.exists(output_path):
#                     opencodedbase64 = encodeImageIntoBase64(output_path)
#                     result = {"image": opencodedbase64.decode('utf-8')}
#                 else:
#                     raise FileNotFoundError(f"Output image not found at {output_path}")
#             else:
#                 raise FileNotFoundError("No valid output directory found")
#         else:
#             raise FileNotFoundError("No 'exp*' directories found in yolov5/runs/detect")

#         # Clean up the runs directory after prediction
#         shutil.rmtree('yolov5/runs', ignore_errors=True)

#     except ValueError as val:
#         print(val)
#         return Response("Value not found inside JSON data")
#     except KeyError:
#         return Response("Key value error: incorrect key passed")
#     except FileNotFoundError as fnf_error:
#         print(fnf_error)
#         return Response(fnf_error)
#     except Exception as e:
#         print(e)
#         result = "Invalid input"

#     return jsonify(result)


# @app.route("/live", methods=['GET'])
# @cross_origin()
# def predictLive():
#     try:
#         os.system("cd yolov5/ && python detect.py --weights best.pt --img 416 --conf 0.5 --source 0")
#         shutil.rmtree('yolov5/runs', ignore_errors=True)
#         return "Camera starting!!"

#     except ValueError as val:
#         print(val)
#         return Response("Value not found inside JSON data")

# if __name__ == "__main__":
#     clApp = ClientApp()
#     app.run(host="0.0.0.0", port=8080)
