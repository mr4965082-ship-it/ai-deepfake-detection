import os
import cv2

from flask import Flask
from flask import render_template
from flask import request

from werkzeug.utils import secure_filename

from utils.detector import predict_image
from utils.detector import predict_video


app = Flask(__name__)


UPLOAD_FOLDER = "static/uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


ALLOWED_IMAGE_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png"
}


ALLOWED_VIDEO_EXTENSIONS = {
    "mp4",
    "avi",
    "mov",
    "mkv"
}


def allowed_image(filename):

    return (
        "." in filename
        and
        filename.rsplit(".", 1)[1].lower()
        in ALLOWED_IMAGE_EXTENSIONS
    )


def allowed_video(filename):

    return (
        "." in filename
        and
        filename.rsplit(".", 1)[1].lower()
        in ALLOWED_VIDEO_EXTENSIONS
    )


@app.route("/")
def home():

    return render_template(
        "index.html"
    )


@app.route(
    "/detect",
    methods=["POST"]
)
def detect():

    image = request.files.get("image")

    video = request.files.get("video")


    # IMAGE DETECTION

    if image and image.filename != "":

        if not allowed_image(image.filename):

            return "Invalid image format"


        filename = secure_filename(
            image.filename
        )


        image_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )


        image.save(image_path)


        image_data = cv2.imread(
            image_path
        )


        if image_data is None:

            return "Unable to read image"


        result = predict_image(
            image_data
        )


        return render_template(
            "result.html",

            filename=filename,

            file_type="Image",

            result=result["result"],

            confidence=result["confidence"]
        )


    # VIDEO DETECTION

    if video and video.filename != "":

        if not allowed_video(video.filename):

            return "Invalid video format"


        filename = secure_filename(
            video.filename
        )


        video_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )


        video.save(video_path)


        result = predict_video(
            video_path
        )


        return render_template(
            "result.html",

            filename=filename,

            file_type="Video",

            result=result["result"],

            confidence=result["confidence"]
        )


    return "Please upload an image or video"


if __name__ == "__main__":

    os.makedirs(
        UPLOAD_FOLDER,
        exist_ok=True
    )


   if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    app.run(
        debug=False,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )