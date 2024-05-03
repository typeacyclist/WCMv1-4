import json
import sys
import threading
import time
import warnings
from datetime import datetime

import dlib
import pandas as pd
# import keyboard
import pyautogui
import torch
from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap
from retinaface import RetinaFace
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
# from serialUSB import runSerial
from GUI import Ui_MainWindow
from ThreadHandler import SimpleThread
from YoloDetector import YOLOV8_Detector
from helperFunctions import *


# Display the torch is avaliable or not
print("Pytorch:", torch.cuda.is_available())

# Remove all the useless Warnings
warnings.filterwarnings('ignore')


# Application class
class Application:

    # Constructor of the application Class
    def __init__(self):

        # Default variable values
        self.status = {}
        self.face_enc = []
        self.fp_boxes = None
        self.grayFrame = None
        self.face_coor = None
        self.faceLocation = []
        self.stream_Image = None
        self.detected_image = None
        self.reference_face = None
        self.cellPhoneInUse = False
        self.cellPhoneInArea = False
        self.stream_Image_copy = None
        self.detected_class_count = {}
        self.cellPhoneTakePics = False
        self.csvLogFileCreated = False
        self.screenShotDetected = False
        self.fingerPrintDeviceCount = 0
        self.reference_image_encoding = []
        self.detected_class_confidence = {}
        self.reference_image_captured = False
        self.detected_classes_coordinates = {}
        self.csv_log_file_save_path = csv_log_file_save_path
        self.objects_to_detect_id_name = original_name_english.copy()
        self.violation_types_id_name = violation_types_id_name_original.copy()
        self.rule_violation_fix_messages = rule_violation_fix_messages_original.copy()
        self.detected_objects_confidence_threshold = detected_objects_confidence_threshold.copy()
        self.violation_types_trigger_action_message = violation_types_trigger_action_message_original.copy()

        # Connecting the backend to the frontend
        self.MainWindow = QtWidgets.QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.MainWindow)

        # Initialize the Spin boxes
        self.initializeObjectConfValues()

        # Thread for the values update
        self.WSAD = SimpleThread()
        self.WSAD.signal.connect(self.flagsUpdator)
        self.WSAD.camera_update_time = 1
        self.WSAD.start()

        # Thread to update the variables
        self.IDV = SimpleThread()
        self.IDV.signal.connect(self.cameraFrameUpdator)
        self.IDV.camera_update_time = 1
        self.IDV.start()

        # Start the detector thread
        self.cameraStreamThread = threading.Thread(target=self.cameraStream)
        self.cameraStreamThread.start()

        # Face detector model
        self.face_detector = RetinaFace()
        self.shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks(1).dat')
        self.face_rec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

        # Language switched
        self.ui.comboBox.currentIndexChanged.connect(self.languageSwitched)

        # if the values of the Spin box is changed
        self.ui.spinBox.valueChanged.connect(self.changeConfidenceValues)
        self.ui.spinBox_2.valueChanged.connect(self.changeConfidenceValues)
        self.ui.spinBox_3.valueChanged.connect(self.changeConfidenceValues)
        self.ui.spinBox_4.valueChanged.connect(self.changeConfidenceValues)
        self.ui.spinBox_5.valueChanged.connect(self.changeConfidenceValues)
        self.ui.spinBox_6.valueChanged.connect(self.changeConfidenceValues)
        self.ui.spinBox_7.valueChanged.connect(self.changeConfidenceValues)

        # Button Connections
        self.ui.pushButton.clicked.connect(self.cameraFrameUpdator)
        self.ui.pushButton_2.clicked.connect(self.refreshReferenceImage)

        # Load the Model of the Detector and finger print
        self.detector = YOLOV8_Detector(weights='yolov8s.pt',
                                        img_size=640,
                                        confidence_thres=0.2,
                                        iou_thresh=0.6,
                                        agnostic_nms=True,
                                        augment=True)
        self.fingerPrintScannerModel = YOLO( 'model/fingerPrint-BAK.pt' )

    # Thread to detect the screenshots
    def screenShotDetect(self):
        self.screenShotDetected = True

    # Set the values in the Gui of confidence at the starting
    def initializeObjectConfValues(self):
        spinBoxes = [self.ui.spinBox, self.ui.spinBox_2, self.ui.spinBox_3,
                     self.ui.spinBox_4, self.ui.spinBox_5, self.ui.spinBox_6, self.ui.spinBox_7]
        labels = ['person', 'chair', 'laptop', 'mouse', 'keyboard', 'cell phone', 'book']

        for label, spinBox in zip(labels, spinBoxes):
            spinBox.setValue(int(float(detected_objects_confidence_threshold[label.lower()]) * 100))

    # Update the Confidence values
    def changeConfidenceValues(self):
        conf_values = [self.ui.spinBox.text(), self.ui.spinBox_2.text(), self.ui.spinBox_3.text(),
                       self.ui.spinBox_4.text(), self.ui.spinBox_5.text(), self.ui.spinBox_5.text(),
                       self.ui.spinBox_6.text(), self.ui.spinBox_7.text()]
        labels = ['person', 'chair', 'laptop', 'mouse', 'keyboard', 'cell phone', 'book']

        for label, conf in zip(labels, conf_values):
            self.detected_objects_confidence_threshold[label.lower()] = int(conf) / 100

    # Function to switch the languages
    def languageSwitched(self):

        # Display the Warnings
        self.displayLeftMessages()

        # Display the left Labels
        self.displayRightMessages()

        # Display the values of the left side classes in their languages
        target_labels_list = [self.ui.label_18, self.ui.label_19, self.ui.label_20, self.ui.label_21,
                              self.ui.label_23, self.ui.label_24, self.ui.label_25,
                              self.ui.label_26, self.ui.label_27, self.ui.label_28, self.ui.label_29]

        if self.ui.comboBox.currentText() == "English":
            self.objects_to_detect_id_name = original_name_english.copy()
            self.violation_types_id_name = violation_types_id_name_original.copy()
            self.rule_violation_fix_messages = rule_violation_fix_messages_original.copy()
            self.violation_types_trigger_action_message = violation_types_trigger_action_message_original.copy()
            for n, l in enumerate(target_labels_list):
                l.setText(rule_id_name[n + 1])
        else:
            self.violation_types_id_name = violation_types_id_name_altlang_chinese.copy()
            self.objects_to_detect_id_name = objects_to_detect_id_name_altlang_chinese.copy()
            self.rule_violation_fix_messages = rule_violation_fix_messages_altlang_chinese.copy()
            self.violation_types_trigger_action_message = violation_types_trigger_action_message_altlang_chinese.copy()
            for n, l in enumerate(target_labels_list):
                l.setText(rule_id_name_altlang_chinese[n + 1])

    # When the application will close
    def stopProcesses(self):
        self.WSAD.stop()
        self.IDV.stop()

    # Perform object detection on the frame
    def detectOnFrame(self):

        results = self.fingerPrintScannerModel(self.stream_Image,
                                               conf=0.3,
                                               device="cpu")

        # Count the detected fingerprint scanners
        self.fingerPrintDeviceCount = 0

        # Get the fingerprint boxes
        self.fp_boxes = []
        for result in results:
            if result:
                x1, y1, x2, y2 = [int(i) for i in result.boxes.data.tolist()[0][:4]]
                self.fp_boxes.append([x1, y1, x2, y2])

        self.fingerPrintDeviceCount = len(self.fp_boxes)

        if self.ui.comboBox.currentText() == "English":
            language = "English"
        else:
            language = "Chinese"

        response = self.detector.Detect(self.stream_Image,
                                        self.objects_to_detect_id_name,
                                        self.detected_objects_confidence_threshold,
                                        detected_objects_bbox_label_line_color,
                                        original_name_english,
                                        self.fp_boxes, language)

        self.detected_image = response[0]
        self.grayFrame = response[1]
        self.detected_class_count = response[2]
        self.detected_classes_coordinates = response[3]
        self.detected_class_confidence = response[4]

    # Function in which the label will be updated
    def cameraStream(self):
        vid = cv2.VideoCapture(camera_number)

        while self.WSAD.thread_Active:
            ret, self.stream_Image = vid.read()

            # If the frame is Captured
            if ret:
                self.stream_Image = cv2.resize(self.stream_Image, (640, 480))
                self.stream_Image_copy = self.stream_Image.copy()
                self.stream_Image = self.stream_Image.copy()

                #time.sleep(2)

    # Function in which it updates all the GUI
    def flagsUpdator(self):

        # Update the detected frame image
        if self.detected_image is not None:
            # Display the Warnings
            self.displayLeftMessages()

            # Display the left Labels
            self.displayRightMessages()

            # Save the state
            self.saveStatusJson()

            # Save the results into the csv
            self.save_in_csv()

            # Save the Frame
            cv2.imwrite("detectionFrame.jpg", self.detected_image)

            # Save the Webcam ScreenShot
            cv2.imwrite(live_image_save_path, self.stream_Image)

    # Function in which it updates all the Qt Values
    def cameraFrameUpdator(self):

        # Update the detected frame image
        if self.stream_Image is not None:

            # Draw on latest frame
            self.detectOnFrame()

            # Check for the rules
            self.checkRules()

            # Update the live Frame
            if sum(self.detected_class_count.values()) != 0:

                detected_frame = self.detected_image.copy()

                # Display the reference image
                if type(self.reference_face) == np.ndarray:
                    self.reference_face = cv2.resize(self.reference_face, (100, 100))
                    detected_frame[-100:, -100:] = self.reference_face.copy()

                # Convert it for the frame
                image = NumpyToQImage(detected_frame)
                self.ui.label_5.setPixmap(QPixmap.fromImage(image))
            else:
                image = NumpyToQImage(self.grayFrame)
                self.ui.label_5.setPixmap(QPixmap.fromImage(image))

            # Change the refresh rate
            self.IDV.camera_update_time = self.ui.doubleSpinBox.value()

            # Save the screenshot
            pyautogui.screenshot().save(screen_shot_path)

    # Function to Check the Quality
    def checkImageQuality(self):
        if sum(self.detected_class_count.values()) == 0:

            # Display the Alert
            display_red_box_alert(self.ui.label_6, self.ui.comboBox.currentText())

            # Make the rest of the labels as empty
            for label in [self.ui.label_7, self.ui.label_8, self.ui.label_9, self.ui.label_11,
                          self.ui.label_12, self.ui.label_13, self.ui.label_14, self.ui.label_15, self.ui.label_16,
                          self.ui.label_17]:
                label.setStyleSheet(f"background-color: rgb{violation_types_rule_status_box_fill_color[1]};"
                                    f"color: rgb{violation_types_rule_status_box_font_color[1]};")
                label.setText("")
            return
        else:
            display_white_box_ok(self.ui.label_6, self.ui.comboBox.currentText())

            # Make the rest of the label as OK
            for label in [self.ui.label_7, self.ui.label_8, self.ui.label_9, self.ui.label_11,
                          self.ui.label_12, self.ui.label_13, self.ui.label_14, self.ui.label_15, self.ui.label_16,
                          self.ui.label_17]:
                label.setStyleSheet(f"background-color: rgb{violation_types_rule_status_box_fill_color[1]};"
                                    f"color: rgb{violation_types_rule_status_box_font_color[1]};")
                label.setText("OK")

    # Function to Check the webcam alignment
    def checkWebCamAlignment(self):
        if self.detected_class_count[0] >= 1 and \
                (self.detected_class_count[63] >= 1 or self.detected_class_count[66] >= 1):
            display_white_box_ok(self.ui.label_7, self.ui.comboBox.currentText())
        else:
            display_red_box_alert(self.ui.label_7, self.ui.comboBox.currentText())

    # Function to check the Identity
    def checkIdentity(self):

        # Check for the person class is detected or not
        if self.detected_class_count[0] >= 1:

            # Get the first person
            Ix1, Iy1, Ix2, Iy2 = self.detected_classes_coordinates[0][0]
            target_person = self.detected_image[Iy1:Iy2, Ix1:Ix2]

            # Extracting the face
            faces = self.face_detector.predict(cv2.cvtColor(target_person, cv2.COLOR_BGR2RGB))

            # Check for the face is detected or not
            if len(faces) != 0:

                # Display the indicator for the face is detected
                display_white_box_ok(self.ui.label_16, self.ui.comboBox.currentText())

                # Get the first face
                face = faces[0]

                # Extract face bounding box coordinates
                x1, y1, x2, y2 = abs(face['x1']) + Ix1, abs(face['y1']) + Iy1, abs(face['x2']) + Ix1, abs(face['y2']) + Iy1

                # Get the center point of the face
                fx = x1 + (x2 - x1) // 2
                fy = y1 + (y2 - y1) // 2

                # Get the face center point
                self.face_coor = (fx, fy)

                # Extract the face Encodings convert face region to dlib rectangle
                rect = dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2)

                # Detect face landmarks
                landmarks = self.shape_predictor(self.detected_image, rect)

                # Generate face encodings using the face recognition model
                self.face_enc = np.array(self.face_rec.compute_face_descriptor(self.detected_image, landmarks))

                # Display face box
                cv2.rectangle(self.detected_image, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)

                # Check if the encodings are extracted or not
                if len(self.face_enc) > 0:

                    # Check if this is the first face
                    if not self.reference_image_captured:

                        # Crop face region
                        face = self.detected_image

                        # Convert the face to RGB for face recognition
                        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                        # Convert face region to dlib rectangle
                        rect = dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2)

                        # Detect face landmarks
                        landmarks = self.shape_predictor(rgb, rect)

                        # Generate face encodings using the face recognition model
                        self.reference_image_encoding = \
                            np.array(self.face_rec.compute_face_descriptor(self.detected_image, landmarks))

                        # Save the reference face
                        self.reference_face = self.stream_Image_copy[y1:y2, x1:x2]
                        self.reference_image_captured = True
                        self.faceLocation = [(x1, y1), (x2, y2)]

                    else:

                        # Calculate the cosine similarity between the encodings
                        similarity = cosine_similarity([self.reference_image_encoding], [self.face_enc])[0][0]

                        # Compare the similarity to the threshold
                        if similarity > face_match_thresh:
                            matched = True
                        else:
                            matched = False

                        # Save the results if matched
                        if matched:
                            self.face_enc = self.face_enc.tolist()
                            self.faceLocation = [(x1, y1), (x2, y2)]
                            display_white_box_ok(self.ui.label_8, self.ui.comboBox.currentText())
                            cv2.rectangle(self.detected_image, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)

                        # Otherwise save nothing
                        else:
                            self.faceLocation = []
                            display_red_box_alert(self.ui.label_8, self.ui.comboBox.currentText())
                            cv2.rectangle(self.detected_image, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
                else:
                    self.faceLocation = []
                    display_red_box_alert(self.ui.label_8, self.ui.comboBox.currentText())
                    cv2.rectangle(self.detected_image, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
                    # runSerial()
            else:
                display_Refer_box(self.ui.label_16, self.ui.comboBox.currentText())
                self.face_coor = None

    # Function to check the observer
    def checkObserver(self):
        if self.detected_class_count[0] > 1:
            display_red_box_alert(self.ui.label_9, self.ui.comboBox.currentText())
        else:
            display_white_box_ok(self.ui.label_9, self.ui.comboBox.currentText())

    # Function to check the Mobile to take phot
    def checkMobileInUseAndTakePhoto(self):
        if self.detected_class_count[0] != 0 and self.detected_class_count[67] != 0 and self.face_coor is not None:

            # Finding the closest person to the center
            closest_person = []
            if self.detected_class_count[0] > 1:
                distance = 99999999999999

                # Finding the center closest person
                for coor in self.detected_classes_coordinates[0]:
                    x1 = coor[0]
                    y1 = coor[1]
                    x2 = coor[2]
                    y2 = coor[3]
                    cx = x1 + (x2 - x1) // 2
                    cy = y1 + (y2 - y1) // 2

                    h, w, _ = self.detected_image.shape
                    sx, sy = w // 2, h // 2

                    if dist([sx, sy], [cx, cy]) < distance:
                        distance = dist([sx, sy], [cx, cy])
                        closest_person = coor
            else:
                closest_person = self.detected_classes_coordinates[0][0]

            # Get the closest person coordinates
            closest_x1 = closest_person[0]
            closest_y1 = closest_person[1]
            closest_x2 = closest_person[2]
            closest_y2 = closest_person[3]

            # check if any mobile is in the closest person bounding box
            for coor in self.detected_classes_coordinates[67]:
                x1 = coor[0]
                y1 = coor[1]
                x2 = coor[2]
                y2 = coor[3]

                # Mobile center
                cx = x1 + ((x2 - x1) // 2)
                cy = y1 + ((y2 - y1) // 2)

                # Showing the warning if the mobile is in the person box
                if closest_x1 < cx < closest_x2 and closest_y1 < cy < closest_y2:
                    self.cellPhoneInUse = True
                    display_red_box_alert(self.ui.label_11, self.ui.comboBox.currentText())
                    # runSerial()
                else:
                    self.cellPhoneInUse = False
                    display_white_box_ok(self.ui.label_11, self.ui.comboBox.currentText())

                # Remove the warning if the mobile is not the fingerPrint box
                for fp_coor in self.fp_boxes:
                    fpx1, fpy1, fpx2, fpy2 = fp_coor
                    if fpx1 < cx < fpx2 and fpy1 < cy < fpy2:
                        self.cellPhoneInUse = False
                        display_white_box_ok(self.ui.label_11, self.ui.comboBox.currentText())


        else:
            display_white_box_ok(self.ui.label_11, self.ui.comboBox.currentText())

    # Check for the Cell in area
    def checkCellInArea(self):
        if self.detected_class_count[67] != 0:
            self.cellPhoneInArea = True
            display_Refer_box(self.ui.label_12, self.ui.comboBox.currentText())
            # runSerial()
        else:
            self.cellPhoneInArea = False
            display_white_box_ok(self.ui.label_12, self.ui.comboBox.currentText())

    # Check for the Multiple keyboard
    def checkMultipleKeyboard(self):
        if self.detected_class_count[66] > 1:
            display_aware_box(self.ui.label_13, self.ui.comboBox.currentText())
        else:
            display_white_box_ok(self.ui.label_13, self.ui.comboBox.currentText())

    # Check the Multiple Paper Books
    def checkMultiplePaperBooks(self):
        if self.detected_class_count[73] > 0:
            display_aware_box(self.ui.label_14, self.ui.comboBox.currentText())
        else:
            display_white_box_ok(self.ui.label_14, self.ui.comboBox.currentText())

    # Check for thr Unattended Person
    def unattendedPerson(self):
        if self.detected_class_count[0] == 0:
            display_aware_box(self.ui.label_15, self.ui.comboBox.currentText())
        else:
            display_white_box_ok(self.ui.label_15, self.ui.comboBox.currentText())

    # Check for the screenShot Captured
    def checkScreenShot(self):
        if self.screenShotDetected:
            display_red_box_alert(self.ui.label_17, self.ui.comboBox.currentText())
            self.screenShotDetected = False
        else:
            display_white_box_ok(self.ui.label_17, self.ui.comboBox.currentText())

    def refreshReferenceImage(self):
        self.reference_image_captured = False

    # Main Function where the system is checking for the Rules
    def checkRules(self):

        # Priority 1 Check for image Quality
        self.checkImageQuality()

        # Priority 2 Webcam Alignment
        self.checkWebCamAlignment()

        # Priority 3 Identity
        self.checkIdentity()

        # Priority 4 Observers Check
        self.checkObserver()

        # Priority 5, 6 Check for the mobile in use and mobile takes pics or not
        self.checkMobileInUseAndTakePhoto()

        # Priority 7 Cell in Area
        self.checkCellInArea()

        # Priority 8 Multiple Keyboards
        self.checkMultipleKeyboard()

        # Priority 9 Multiple Paper Book
        self.checkMultiplePaperBooks()

        # Priority 10 Unattended Person
        self.unattendedPerson()

        # Priority 12 Check for the ScreenShot detect
        self.checkScreenShot()

    # Save the current Status
    def saveStatusJson(self):

        if self.stream_Image is None:
            return

        # Update the status with the date and time
        self.status = {"captureDate": datetime.now().strftime("%y:%m:%d"),
                       "captureTime": datetime.now().strftime("%H:%M:%S"), "imageQualityTest": {}}

        # Update the Image Quality test
        if calculate_focal_measure(self.stream_Image) > 150:
            self.status["imageQualityTest"]['Blurry'] = True
        else:
            self.status["imageQualityTest"]['Blurry'] = False

        # Update Cell Phone Flags
        self.status["CellPhoneInUse"] = self.cellPhoneInUse
        self.status["CellPhoneInArea"] = self.cellPhoneInArea
        self.status["CellPhoneTakePics"] = self.cellPhoneTakePics

        # Update the detected objects Metadata
        self.status["detectedObjectsCount"] = {}
        self.status["detectedObjectConfidence"] = {}
        for i in self.detected_class_count.keys():
            self.status["detectedObjectsCount"][self.objects_to_detect_id_name[i]] = \
                self.detected_class_count[i]
            self.status["detectedObjectConfidence"][self.objects_to_detect_id_name[i]] = \
                self.detected_class_confidence[i]

        # Update the face Information
        self.status["faceInfo"] = {}

        if len(self.face_enc) > 0:
            self.status["faceInfo"]["faceDetect"] = True

            if type(self.face_enc) == np.ndarray:
                self.face_enc = self.face_enc.tolist()

            self.status["faceInfo"]["faceEncodings"] = self.face_enc
            self.status["faceInfo"]["faceLocation"] = self.faceLocation
        else:
            self.status["faceInfo"]["faceDetect"] = False
            self.status["faceInfo"]["faceEncodings"] = []
            self.status["faceInfo"]["faceLocation"] = []

        # Update the Object type counts
        self.status["ObjectTypeCount"] = {}
        self.status["ObjectTypeCount"]["personObjects"] = {}
        self.status["ObjectTypeCount"]["keyboardObjects"] = {}
        self.status["ObjectTypeCount"]["deskObjects"] = {}

        for i in [0, 56]:
            self.status["ObjectTypeCount"]["personObjects"][self.objects_to_detect_id_name[i]] = \
                self.detected_class_count[i]
        for i in [63, 66]:
            self.status["ObjectTypeCount"]["keyboardObjects"][self.objects_to_detect_id_name[i]] = \
                self.detected_class_count[i]
        for i in [63, 64, 66]:
            self.status["ObjectTypeCount"]["deskObjects"][self.objects_to_detect_id_name[i]] = \
                self.detected_class_count[i]

        # Observers information
        if self.ui.label_19.text() == "OK":
            self.status["observer"] = False
            self.status["observerCount"] = 0
        else:
            self.status["observer"] = True
            self.status["observerCount"] = self.detected_class_count[0]

        # Save the finger print count
        self.status["FPS"] = self.fingerPrintDeviceCount

        print("Finger print Count:", self.fingerPrintDeviceCount)

        # Save the json file
        json.dump(self.status, open("status.json", "w", encoding='utf-8'), indent=4)

    # Function to save the status in the csv file
    def save_in_csv(self):
        if not self.csvLogFileCreated:
            self.csv_log_file_save_path = datetime.now().strftime("%y%m%d%H%M%S") + "Status_CSV.csv"
            status_csv = pd.DataFrame([self.status])
            status_csv.to_csv(self.csv_log_file_save_path, index=False)
            self.csvLogFileCreated = True
        else:
            status_csv = pd.read_csv(self.csv_log_file_save_path)
            status_csv = pd.concat([status_csv, pd.DataFrame([self.status])])
            status_csv.to_csv(self.csv_log_file_save_path, index=False)

    # Display Left Message
    def displayLeftMessages(self):

        if self.ui.comboBox.currentText() == "English":
            self.ui.label_34.setText("Objects Confidence Values")
            self.ui.pushButton_2.setText("Update Ref Image")
            self.ui.pushButton.setText("Update Frame")
            self.ui.label_49.setText("REFRESH RATE")
            self.ui.label_33.setText("LANGUAGE")
            self.ui.label_50.setText("Sec's")
        else:
            self.ui.pushButton_2.setText("更新参考图像")
            self.ui.label_34.setText("对象置信度值")
            self.ui.pushButton.setText("更新框架")
            self.ui.label_49.setText("刷新率")
            self.ui.label_50.setText("秒的")
            self.ui.label_33.setText("语言")

        # Take all the current Indicating flags
        flags = [self.ui.label_6, self.ui.label_7, self.ui.label_8, self.ui.label_9,
                 self.ui.label_11, self.ui.label_12, self.ui.label_13, self.ui.label_14, self.ui.label_15,
                 self.ui.label_16, self.ui.label_17]

        # Take all the messages of the flags
        flags_name = [self.ui.label_18, self.ui.label_19, self.ui.label_20, self.ui.label_21,
                      self.ui.label_23, self.ui.label_24, self.ui.label_25, self.ui.label_26, self.ui.label_27,
                      self.ui.label_28, self.ui.label_29]

        # Count the total violations
        violations_count = 0

        for i in flags:
            if self.ui.comboBox.currentText() == "English":
                if i.text() in ["ALERT", "WARN", "REFER", "AWARE"]:
                    violations_count += 1
            else:
                if i.text() in ["ALERT", "WARN", "REFER", "AWARE"]:
                    violations_count += 1
        if self.ui.comboBox.currentText() == "English":
            violation_msg = "violation count: " + str(violations_count)
        else:
            violation_msg = "违规次数: " + str(violations_count)

        for msg_number, label, name in zip(range(1, len(flags) + 1), flags, flags_name):

            if label.text() == "ALERT" or label.text() == violation_types_id_name_altlang_chinese[1]:
                # WarningSoundHLH()
                leftAlertMessages(upper_label_1=self.ui.label,
                                  upper_label_2=self.ui.label_2,
                                  lower_label_1=self.ui.label_3,
                                  lower_label_2=self.ui.label_4,
                                  upper_message_1=self.violation_types_id_name[2],
                                  upper_message_2=self.rule_violation_fix_messages[msg_number],
                                  lower_message_1=violation_msg,
                                  lower_message_2=self.violation_types_trigger_action_message[2])
                break

            elif label.text() == "WARN" or label.text() == violation_types_id_name_altlang_chinese[3]:
                # WarningSoundLHL()
                leftWarningMessages(upper_label_1=self.ui.label,
                                    upper_label_2=self.ui.label_2,
                                    lower_label_1=self.ui.label_3,
                                    lower_label_2=self.ui.label_4,
                                    upper_message_1=self.violation_types_id_name[3],
                                    upper_message_2=self.rule_violation_fix_messages[msg_number],
                                    lower_message_1=violation_msg,
                                    lower_message_2=self.violation_types_trigger_action_message[3])
                break

            elif label.text() == "AWARE" or label.text() == violation_types_id_name_altlang_chinese[4]:
                leftAwareMessages(upper_label_1=self.ui.label,
                                  upper_label_2=self.ui.label_2,
                                  lower_label_1=self.ui.label_3,
                                  lower_label_2=self.ui.label_4,
                                  upper_message_1=self.violation_types_id_name[4],
                                  upper_message_2=self.rule_violation_fix_messages[msg_number],
                                  lower_message_1=violation_msg,
                                  lower_message_2=self.violation_types_trigger_action_message[4])

                break

            elif label.text() == "REFER" or label.text() == violation_types_id_name_altlang_chinese[5]:
                leftReferMessages(upper_label_1=self.ui.label,
                                  upper_label_2=self.ui.label_2,
                                  lower_label_1=self.ui.label_3,
                                  lower_label_2=self.ui.label_4,
                                  upper_message_1=self.violation_types_id_name[5],
                                  upper_message_2=self.rule_violation_fix_messages[msg_number],
                                  lower_message_1=violation_msg,
                                  lower_message_2=self.violation_types_trigger_action_message[5])
                break

            else:
                leftOkMessages(upper_label_1=self.ui.label,
                               upper_label_2=self.ui.label_2,
                               lower_label_1=self.ui.label_3,
                               lower_label_2=self.ui.label_4,
                               upper_message_1=self.violation_types_id_name[1],
                               upper_message_2="",
                               lower_message_1=violation_msg,
                               lower_message_2=self.violation_types_trigger_action_message[1])

    # Display the Right Message
    def displayRightMessages(self):
        classConfLabels = [self.ui.label_35, self.ui.label_36, self.ui.label_37, self.ui.label_38, self.ui.label_39,
                           self.ui.label_40, self.ui.label_41]

        if self.ui.comboBox.currentText() == "English":
            for label, name in zip(classConfLabels, objects_to_detect_id_name.keys()):
                label.setText(objects_to_detect_id_name[name].upper())

        else:
            for label, name in zip(classConfLabels, objects_to_detect_id_name_altlang_chinese.keys()):
                label.setText(objects_to_detect_id_name_altlang_chinese[name])


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    # Starting and displaying the window
    application = Application()
    application.MainWindow.show()
    app.exec_()

    # Stop all the thread after the application close

    application.WSAD.stop()
    application.IDV.stop()
    application.cameraStreamThread.join(timeout=1)
    sys.exit(0)
