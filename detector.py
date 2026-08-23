import cv2
import numpy as np

from tensorflow.keras.models import load_model


MODEL_PATH = "model/deepfake_model.h5"

model = load_model(MODEL_PATH)


def preprocess_image(image):

    image = cv2.resize(
        image,
        (128, 128)
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = image.astype(
        "float32"
    ) / 255.0

    image = np.expand_dims(
        image,
        axis=0
    )

    return image


def predict_image(image):

    processed_image = preprocess_image(image)

    prediction = model.predict(
        processed_image,
        verbose=0
    )[0][0]

    if prediction >= 0.5:

        result = "REAL"
        confidence = prediction * 100

    else:

        result = "DEEPFAKE"
        confidence = (1 - prediction) * 100

    return {
        "result": result,
        "confidence": round(
            float(confidence),
            2
        )
    }


def predict_video(video_path):

    cap = cv2.VideoCapture(video_path)

    predictions = []
    frame_count = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_count += 1

        if frame_count % 10 != 0:
            continue

        processed_frame = preprocess_image(frame)

        prediction = model.predict(
            processed_frame,
            verbose=0
        )[0][0]

        predictions.append(
            float(prediction)
        )

    cap.release()

    if len(predictions) == 0:

        return {
            "result": "Unable to analyze video",
            "confidence": 0
        }

    average_prediction = np.mean(
        predictions
    )

    if average_prediction >= 0.5:

        result = "DEEPFAKE"
        confidence = average_prediction * 100

    else:

        result = "REAL"
        confidence = (1 - average_prediction) * 100

    return {
        "result": result,
        "confidence": round(
            float(confidence),
            2
        )
    }