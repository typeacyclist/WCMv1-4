import math

import cv2
import numpy as np
# import winsound
from PyQt5 import QtGui
from PyQt5.QtGui import QImage
from Parameters import *


def get_embedding(emb_model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = np.expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = emb_model.predict(samples)
    return yhat[0]


def mobilePredict(model, image):
    image = cv2.resize(image, (224, 224))
    image = np.array(image, dtype=np.float32)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    index = np.argmax(prediction)
    return index


# Function between two points
def dist(p1, p2):
    return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


# Method for the conversion from QImage to Numpy array
def QImageToNumpy(qimage):
    qimage = qimage.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
    width = qimage.width()
    height = qimage.height()

    ptr = qimage.bits()
    ptr.setsize(height * width * 4)
    arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
    return arr


# Method to convert the numpy image to QImage
def NumpyToQImage(npImage):
    result = cv2.cvtColor(npImage, cv2.COLOR_BGR2RGB)
    height, width, channel = result.shape
    step = channel * width
    qImg = QImage(result.data, width, height, step, QImage.Format_RGB888)
    return qImg


# Display the warning box
def display_red_box_alert(label, lang):
    label.setStyleSheet(f"background-color: rgb{violation_types_rule_status_box_fill_color[2]};"
                        f"color: rgb{violation_types_rule_status_box_font_color[2]};")
    if lang == "Chinese":
        label.setText("警报")
    else:
        label.setText(violation_types_id_name[2])


# Display the ok box
def display_white_box_ok(label, lang):
    label.setStyleSheet(f"background-color: rgb{violation_types_rule_status_box_fill_color[1]};"
                        f"color: rgb{violation_types_rule_status_box_font_color[1]};")
    if lang == "Chinese":
        label.setText("好的")
    else:
        label.setText("OK")


# Display the warning box
def display_warning_box(label, lang):
    label.setStyleSheet(f"background-color: rgb{violation_types_rule_status_box_fill_color[3]};"
                        f"color: rgb{violation_types_rule_status_box_font_color[3]};")
    if lang == "Chinese":
        label.setText("警告")
    else:
        label.setText("WARN")


# Display the Refer box
def display_Refer_box(label, lang):
    label.setStyleSheet(f"background-color: rgb{violation_types_rule_status_box_fill_color[5]};"
                        f"color: rgb{violation_types_rule_status_box_font_color[5]};")
    if lang == "Chinese":
        label.setText("参考")
    else:
        label.setText("REFER")


# Display the Aware
def display_aware_box(label, lang):
    label.setStyleSheet(f"background-color: rgb{violation_types_rule_status_box_fill_color[4]};"
                        f"color: rgb{violation_types_rule_status_box_font_color[4]};")
    if lang == "Chinese":
        label.setText("知道的")
    else:
        label.setText("AWARE")


def leftOkMessages(upper_label_1,
                   upper_label_2,
                   lower_label_1,
                   lower_label_2,
                   upper_message_1="",
                   upper_message_2="",
                   lower_message_1="",
                   lower_message_2=""):
    upper_label_1.setStyleSheet(f"background-color: rgb{violation_types_rule_status_box_fill_color[6]};"
                                f"color: rgb{violation_types_rule_status_box_font_color[2]};")
    upper_label_1.setText(upper_message_1)

    upper_label_2.setStyleSheet(f"background-color: rgb{violation_types_rule_status_box_fill_color[6]};"
                                f"color: rgb{violation_types_rule_status_box_font_color[2]};")
    upper_label_2.setText(upper_message_2)

    lower_label_1.setStyleSheet(f"background-color: rgb{violation_types_rule_status_box_fill_color[6]};"
                                f"color: rgb{violation_types_rule_status_box_font_color[2]};")
    lower_label_1.setText(lower_message_1)

    lower_label_2.setStyleSheet(f"background-color: rgb{violation_types_rule_status_box_fill_color[6]};"
                                f"color: rgb{violation_types_rule_status_box_font_color[2]};")
    lower_label_2.setText(lower_message_2)


def leftWarningMessages(upper_label_1,
                        upper_label_2,
                        lower_label_1,
                        lower_label_2,
                        upper_message_1="",
                        upper_message_2="",
                        lower_message_1="",
                        lower_message_2=""):
    upper_label_1.setStyleSheet(f"background-color: rgb{violation_types_rule_status_box_fill_color[3]};"
                                f"color: rgb{violation_types_rule_status_box_font_color[2]};")
    upper_label_1.setText(upper_message_1)

    upper_label_2.setStyleSheet(f"background-color: rgb{violation_types_rule_status_box_fill_color[3]};"
                                f"color: rgb{violation_types_rule_status_box_font_color[2]};")
    upper_label_2.setText(upper_message_2)

    lower_label_1.setStyleSheet(f"background-color: rgb{violation_types_rule_status_box_fill_color[3]};"
                                f"color: rgb{violation_types_rule_status_box_font_color[2]};")
    lower_label_1.setText(lower_message_1)

    lower_label_2.setStyleSheet(f"background-color: rgb{violation_types_rule_status_box_fill_color[3]};"
                                f"color: rgb{violation_types_rule_status_box_font_color[2]};")
    lower_label_2.setText(lower_message_2)


def leftAlertMessages(upper_label_1,
                      upper_label_2,
                      lower_label_1,
                      lower_label_2,
                      upper_message_1="",
                      upper_message_2="",
                      lower_message_1="",
                      lower_message_2=""):
    upper_label_1.setStyleSheet(f"background-color: rgb{violation_types_rule_status_box_fill_color[2]};"
                                f"color: rgb{violation_types_rule_status_box_font_color[2]};")
    upper_label_1.setText(upper_message_1)

    upper_label_2.setStyleSheet(f"background-color: rgb{violation_types_rule_status_box_fill_color[2]};"
                                f"color: rgb{violation_types_rule_status_box_font_color[2]};")
    upper_label_2.setText(upper_message_2)

    lower_label_1.setStyleSheet(f"background-color: rgb{violation_types_rule_status_box_fill_color[2]};"
                                f"color: rgb{violation_types_rule_status_box_font_color[2]};")
    lower_label_1.setText(lower_message_1)

    lower_label_2.setStyleSheet(f"background-color: rgb{violation_types_rule_status_box_fill_color[2]};"
                                f"color: rgb{violation_types_rule_status_box_font_color[2]};")
    lower_label_2.setText(lower_message_2)


def leftAwareMessages(upper_label_1,
                      upper_label_2,
                      lower_label_1,
                      lower_label_2,
                      upper_message_1="",
                      upper_message_2="",
                      lower_message_1="",
                      lower_message_2=""):
    upper_label_1.setStyleSheet(f"background-color: rgb{violation_types_rule_status_box_fill_color[4]};"
                                f"color: rgb{violation_types_rule_status_box_font_color[4]};")
    upper_label_1.setText(upper_message_1)

    upper_label_2.setStyleSheet(f"background-color: rgb{violation_types_rule_status_box_fill_color[4]};"
                                f"color: rgb{violation_types_rule_status_box_font_color[4]};")
    upper_label_2.setText(upper_message_2)

    lower_label_1.setStyleSheet(f"background-color: rgb{violation_types_rule_status_box_fill_color[4]};"
                                f"color: rgb{violation_types_rule_status_box_font_color[4]};")
    lower_label_1.setText(lower_message_1)

    lower_label_2.setStyleSheet(f"background-color: rgb{violation_types_rule_status_box_fill_color[4]};"
                                f"color: rgb{violation_types_rule_status_box_font_color[4]};")
    lower_label_2.setText(lower_message_2)


def leftReferMessages(upper_label_1,
                      upper_label_2,
                      lower_label_1,
                      lower_label_2,
                      upper_message_1="",
                      upper_message_2="",
                      lower_message_1="",
                      lower_message_2=""):
    upper_label_1.setStyleSheet(f"background-color: rgb{violation_types_rule_status_box_fill_color[5]};"
                                f"color: rgb{violation_types_rule_status_box_font_color[5]};")
    upper_label_1.setText(upper_message_1)

    upper_label_2.setStyleSheet(f"background-color: rgb{violation_types_rule_status_box_fill_color[5]};"
                                f"color: rgb{violation_types_rule_status_box_font_color[5]};")
    upper_label_2.setText(upper_message_2)

    lower_label_1.setStyleSheet(f"background-color: rgb{violation_types_rule_status_box_fill_color[5]};"
                                f"color: rgb{violation_types_rule_status_box_font_color[5]};")
    lower_label_1.setText(lower_message_1)

    lower_label_2.setStyleSheet(f"background-color: rgb{violation_types_rule_status_box_fill_color[5]};"
                                f"color: rgb{violation_types_rule_status_box_font_color[5]};")
    lower_label_2.setText(lower_message_2)


# def WarningSoundHLH():
#     winsound.Beep(2500, 100)
#     winsound.Beep(1000, 100)
#     winsound.Beep(2500, 100)
#
#
# def WarningSoundLHL():
#     winsound.Beep(1000, 100)
#     winsound.Beep(2500, 100)
#     winsound.Beep(1000, 100)


def calculate_focal_measure(img):
    # convert RGB image to Gray scale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Measure focal measure score (laplacian approach)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm


def calculate_distance(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance
