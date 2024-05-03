import cv2
import pandas as pd
import torch
import numpy as np
from Parameters import *
from ultralytics import YOLO
from helperFunctions import dist
from PIL import Image, ImageFont, ImageDraw


def draw_box_string(img, string, org, thickness):
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font="chinese.simhei.ttf", size=thickness * 10, encoding="unic")
    draw.text(xy=(org[0], org[1] - 10), text=string, fill=(0, 0, 0), font=font)
    img = np.asarray(img)
    cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness

    # Draw the rectangle
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA)

    # Add the label of the object
    tf = max(tl - 1, 1)  # font thickness
    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)

    if label in ["person", "backpack", "handbag", "cup", "fork", "knife", "spoon",
                 "bowl", "chair", "laptop", "mouse", "fps", "keyboard", "finger print", "cell phone", "book"]:
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)
    else:
        img = draw_box_string(img, label, (c1[0], c1[1] - 2), tf)

    return img


# def getSortedPersonList(img0, pred):
#     # Get the image screen center x and y
#     fh, fw = img0.shape[0] // 2, img0.shape[1] // 2
#
#     # Extract the person boxes distance from center
#     person_boxes = []
#     distances = []
#     confidences = []
#
#     # Get the coordinates of the boxes and distance from the center
#     for result in pred:
#         if result:
#             for res in result.boxes.data.tolist():
#                 if 0 == res[5]:
#                     coordinates = res[:4]
#                     cx1, cy1, cx2, cy2 = [int(i) for i in coordinates]
#
#                     cx = cx1 + (cx2 - cx1) // 2
#                     cy = cy1 + (cy2 - cy1) // 2
#
#                     d = dist([cx, cy], [fh, fw])
#                     person_boxes.append([(cx1, cy1), (cx2, cy2)])
#                     distances.append(d)
#                     confidences.append(res[4])
#
#     # Save the distance, coordinates and confidences
#     df = pd.DataFrame()
#     df["distances"] = distances
#     df["coordinates"] = person_boxes
#     df["confidence"] = confidences
#     df = df.sort_values(by="distances", ascending=True)
#
#     return df


class YOLOV8_Detector:
    def __init__(self, weights, img_size, confidence_thres, iou_thresh, agnostic_nms, augment):
        self.weights = weights
        self.imgsz = img_size
        self.conf_thres = confidence_thres
        self.iou_thres = iou_thresh

        self.agnostic_nms = agnostic_nms
        self.augment = augment

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(weights)

    def Detect(self, img0, objects_to_detect_id_name,
               detected_objects_confidence_threshold,
               detected_objects_bbox_label_line_color,
               original_name_english,
               fp_boxes, language):

        # Make the class count, confidence and coordinates dictionary
        detected_class_count = {}
        detected_class_confidence = {}
        detected_classes_coordinates = {}

        # Get the prediction from the image
        pred = self.model(img0,
                          show_conf=False,
                          show_labels=False,
                          classes=[0, 63, 66, 67, 73],
                          iou=self.iou_thres,
                          augment=self.augment,
                          conf=self.conf_thres,
                          agnostic_nms=self.agnostic_nms)

        # Get the image screen center x and y
        fh, fw = img0.shape[0] // 2, img0.shape[1] // 2

        # Extract the person boxes distance from center
        person_boxes = []
        distances = []
        confidences = []
        max_dist = 999999

        # Get the coordinates of the boxes and distance from the center
        for result in pred:
            if result:
                for res in result.boxes.data.tolist():
                    if 0 == res[5]:
                        coordinates = res[:4]
                        cx1, cy1, cx2, cy2 = [int(i) for i in coordinates]

                        cx = cx1 + (cx2 - cx1) // 2
                        cy = cy1 + (cy2 - cy1) // 2

                        d = dist([cx, cy], [fh, fw])

                        if d < max_dist:
                            max_dist = d
                            person_boxes.append([(cx1, cy1), (cx2, cy2)])
                            distances.append(d)
                            confidences.append(res[4])

        # Save the distance, coordinates and confidences
        df = pd.DataFrame()
        df["distances"] = distances
        df["coordinates"] = person_boxes
        df["confidence"] = confidences
        df = df.sort_values(by="distances", ascending=True)

        # Initialize the values for every
        for key in objects_to_detect_id_name:
            detected_class_count[key] = 0
            detected_class_confidence[key] = 0
            detected_classes_coordinates[key] = []

        current_person = 0

        for value in df.values:
            x1 = value[1][0][0]
            y1 = value[1][0][1]
            x2 = value[1][1][0]
            y2 = value[1][1][1]

            # Save the count of the class
            detected_class_count[0] += 1

            # Save the confidence values
            detected_class_confidence[0] = detected_objects_confidence_threshold[original_name_english[0]]

            # Save each class coordinates
            detected_classes_coordinates[0].append([int(x1), int(y1), int(x2), int(y2)])

            # Increment the count
            current_person += 1

            # Check if this is the first one person detected
            if current_person > 1:

                # Check if the person text is in english
                if objects_to_detect_id_name[0] == "person":

                    # Plot the box with observers color
                    img0 = plot_one_box([x1, y1, x2, y2],
                                        img0,
                                        color=detected_objects_bbox_label_line_color["observer"],
                                        label="observer: " + str(current_person),
                                        line_thickness=line_thickness)
                else:

                    # Plot the box with observers color
                    img0 = plot_one_box([x1, y1, x2, y2],
                                        img0,
                                        color=detected_objects_bbox_label_line_color["observer"],
                                        label="观察者: " + str(current_person),
                                        line_thickness=line_thickness)

            else:
                # Plot the box with person color
                img0 = plot_one_box([x1, y1, x2, y2],
                                    img0,
                                    color=detected_objects_bbox_label_line_color[original_name_english[0]],
                                    label=objects_to_detect_id_name[0],
                                    line_thickness=line_thickness)

        # print("person count:", current_person)

        blur = img0.copy()
        blur = cv2.blur(blur, (50, 50))
        boxes_list = [np.array(i.boxes.data.tolist()) for i in pred]

        # Process detections
        for i, det in enumerate(boxes_list):  # detections per image

            # Check for the detections
            if len(det):

                # Rescale boxes from img_size to im0 size
                det[:, :4] = det[:, :4].round()

                # Check for the count of the person
                # person_count = len([int(i) for i in det[:, -1] if int(i) == 0])

                # Iterating over all the detections
                for *xyxy, conf, cls in reversed(det):

                    # Check to ignore the unusefull classes
                    if int(cls) in list(objects_to_detect_id_name.keys()):

                        # Check for the confidence threshold values
                        if conf >= detected_objects_confidence_threshold[original_name_english[int(cls)]]:

                            # Skip for the person class
                            if int(cls) == 0:
                                continue

                            # Save the count of the class
                            detected_class_count[int(cls)] += 1

                            # Save the confidence values
                            detected_class_confidence[int(cls)] = detected_objects_confidence_threshold[original_name_english[int(cls)]]

                            # Save each class coordinates
                            detected_classes_coordinates[int(cls)].append([int(i) for i in xyxy])

                            # Check for the class is of person or not
                            # if int(cls) == 0:
                            #
                            #     # Increment the count
                            #     current_person += 1
                            #
                            #     # Check if this is the first one person detected
                            #     if current_person != 1:
                            #
                            #         # Check if the person text is in english
                            #         if objects_to_detect_id_name[int(cls)] == "person":
                            #
                            #             # Plot the box with observers color
                            #             img0 = plot_one_box(xyxy,
                            #                                 img0,
                            #                                 color=detected_objects_bbox_label_line_color["observer"],
                            #                                 label="observer",
                            #                                 line_thickness=line_thickness)
                            #         else:
                            #
                            #             # Plot the box with observers color
                            #             img0 = plot_one_box(xyxy,
                            #                                 img0,
                            #                                 color=detected_objects_bbox_label_line_color["observer"],
                            #                                 label="观察者",
                            #                                 line_thickness=line_thickness)
                            #
                            #     else:
                            #         # Plot the box with person color
                            #         img0 = plot_one_box(xyxy,
                            #                             img0,
                            #                             color=detected_objects_bbox_label_line_color[
                            #                                 original_name_english[int(cls)]],
                            #                             label=objects_to_detect_id_name[int(cls)],
                            #                             line_thickness=line_thickness)

                            # Check if the bounding boxes overlap with the FPS
                            if int(cls) == 67 and len(fp_boxes) != 0:
                                cx1, cy1, cx2, cy2 = [int(i) for i in xyxy]
                                cx = cx1 + (cx2 - cx1) // 2
                                cy = cy1 + (cy2 - cy1) // 2

                                # if mobile is within any of the fingerprint box
                                for fpb in fp_boxes:
                                    x1, y1, x2, y2 = fpb
                                    if x1 < cx < x2 and y1 < cy < y2:

                                        if language == "English":

                                            # Plot the box with its corresponding colors
                                            img0 = plot_one_box((x1, y1, x2, y2),
                                                                img0,
                                                                color=colors_for_tracking['c.silver'],
                                                                label="fps",
                                                                line_thickness=line_thickness)
                                        else:

                                            # Plot the box with its corresponding colors
                                            img0 = plot_one_box((x1, y1, x2, y2),
                                                                img0,
                                                                color=colors_for_tracking['c.silver'],
                                                                label="指纹",
                                                                line_thickness=line_thickness)
                                    else:
                                        # Plot the box with its corresponding colors
                                        img0 = plot_one_box(xyxy,
                                                            img0,
                                                            color=detected_objects_bbox_label_line_color[
                                                                original_name_english[int(cls)]],
                                                            label=objects_to_detect_id_name[int(cls)],
                                                            line_thickness=line_thickness)
                            else:
                                # Plot the box with its corresponding colors
                                img0 = plot_one_box(xyxy,
                                                    img0,
                                                    color=detected_objects_bbox_label_line_color[
                                                        original_name_english[int(cls)]],
                                                    label=objects_to_detect_id_name[int(cls)],
                                                    line_thickness=line_thickness)

        return [img0, blur, detected_class_count, detected_classes_coordinates, detected_class_confidence]
