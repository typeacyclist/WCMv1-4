
# Number of the camera connected for input stream
camera_number = 0

# Face matching thresh
face_match_thresh = 0.92

# Detection line thickness
line_thickness = 3
fontsize = 40

# YOLO model limits
yolo_model_limit = 2
finger_print_limit = 2
retina_netlimit = 2

# Camera frame update Time in seconds
camera_update_time = 10

# Live Camera Image saving Name
live_image_save_path = "live_frame.jpg"

# Overall screenshot path
screen_shot_path = "screen_Shot.jpg"

# Path to save the csv file
csv_log_file_save_path = "csv_log_file.csv"

# Object Names with their Ids
objects_to_detect_id_name = {
    0: "person",
    24: "backpack",
    26: "handbag",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    56: "chair",
    63: "laptop",
    64: "mouse",
    66: "keyboard",
    67: "cell phone",
    73: "book"
}

# Object Names with their Ids for Recovering
original_name_english = {
    0: "person",
    24: "backpack",
    26: "handbag",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    56: "chair",
    63: "laptop",
    63: "keyboard",
    64: "mouse",
    66: "keyboard",
    67: "cell phone",
    73: "book"
}

# Color over the frames in BGR Format
colors_for_tracking = {
    "c.black": (0, 0, 0),
    "c.blue": (230, 216, 173),
    "c.cyan": (255, 255, 0),
    "c.green": (0, 255, 0),
    "c.grey": (128, 128, 128),
    "c.indigo": (130, 0, 75),
    "c.lime": (0, 255, 0),
    "c.magenta": (255, 0, 255),
    "c.maroon": (0, 0, 128),
    "c.navy": (128, 0, 0),
    "c.olive": (0, 128, 128),
    "c.orange": (0, 165, 255),
    "c.purple": (128, 0, 128),
    "c.red": (0, 0, 255),
    "c.silver": (192, 192, 192),
    "c.teal": (128, 128, 0),
    "c.violet": (238, 130, 238),
    "c.white": (255, 255, 255),
    "c.yellow": (0, 255, 255)
}

# Colors for the GUI in RGB Format
colors_for_gui = {
    "c.black": (0, 0, 0),
    "c.blue": (255, 0, 0),
    "c.cyan": (255, 255, 0),
    "c.green": (0, 130, 124),
    "c.grey": (128, 128, 128),
    "c.indigo": (130, 0, 75),
    "c.lime": (0, 255, 0),
    "c.magenta": (255, 0, 255),
    "c.maroon": (0, 0, 128),
    "c.navy": (128, 0, 0),
    "c.olive": (0, 128, 128),
    "c.orange": (255, 192, 0),
    "c.purple": (128, 0, 128),
    "c.red": (255, 0, 0),
    "c.silver": (192, 192, 192),
    "c.teal": (128, 128, 0),
    "c.violet": (238, 130, 238),
    "c.white": (255, 255, 255),
    "c.yellow": (255, 255, 0)
}

# Object I'd Name for Chinese Language
objects_to_detect_id_name_altlang_chinese = {
    0: "人",
    # 24: "背包",
    # 26: "錢包",
    # 41: "杯子",
    # 42: "叉子",
    # 43: "刀",
    # 44: "勺子",
    # 45: "碗",
    56: "椅子",
    63: "筆記本電腦",
    64: "電腦鼠標",
    66: "鍵盤",
    67: "手機",
    73: "紙"
}

# Detected Object Id names with their Confidence
detected_objects_confidence_threshold = {
    "person": 0.59,
    "backpack": 0.21,
    "handbag": 0.21,
    "cup": 0.21,
    "fork": 0.21,
    "knife": 0.21,
    "spoon": 0.21,
    "bowl": 0.21,
    "chair": 0.21,
    "laptop": 0.21,
    "keyboard": 0.31,
    "mouse": 0.21,
    "cell phone": 0.21,
    "book": 0.11
}

# Detected Object Label Colors
detected_objects_bbox_label_line_color = {
    "person": colors_for_tracking["c.green"],
    "observer": colors_for_tracking["c.orange"],
    "laptop": colors_for_tracking["c.blue"],
    "mouse": colors_for_tracking["c.silver"],
    "chair": colors_for_tracking["c.silver"],
    "keyboard": colors_for_tracking["c.blue"],
    "cell phone": colors_for_tracking["c.yellow"],
    "book": colors_for_tracking["c.violet"],
    "backpack": colors_for_tracking["c.silver"],
    "handbag": colors_for_tracking["c.silver"],
    "cup": colors_for_tracking["c.silver"],
    "fork": colors_for_tracking["c.silver"],
    "knife": colors_for_tracking["c.silver"],
    "spoon": colors_for_tracking["c.silver"],
    "bowl": colors_for_tracking["c.silver"]
}

violation_types_id_name_original = {
    1: "CLEAR",
    2: "ALERT",
    3: "WARN",
    4: "AWARE",
    5: "REFER"
}

violation_types_id_name = {
    1: "CLEAR",
    2: "ALERT",
    3: "WARN",
    4: "AWARE",
    5: "REFER"
}

violation_types_id_name_altlang_chinese = {
    1: "好的",
    2: "警报",
    3: "警告",
    4: "知道的",
    5: "参考"
}

violation_types_rule_status_box_fill_color = {
    1: colors_for_gui["c.white"],
    2: colors_for_gui["c.red"],
    3: colors_for_gui["c.orange"],
    4: colors_for_gui["c.yellow"],
    5: colors_for_gui["c.purple"],
    6: colors_for_gui["c.green"]
}

violation_types_rule_status_box_font_color = {
    1: colors_for_gui["c.black"],
    2: colors_for_gui["c.white"],
    3: colors_for_gui["c.white"],
    4: colors_for_gui["c.black"],
    5: colors_for_gui["c.white"]
}

violation_types_trigger_action_message_original = {
    1: "Log Only",
    2: "Blur > Beep > Notify > Email > Webhook > Log",
    3: "Blur > Beep > Notify > Email > Log",
    4: "Notify > Email > Log",
    5: "Email > Log"
}

violation_types_trigger_action_message = {
    1: "Log Only",
    2: "Blur > Beep > Notify > Email > Webhook > Log",
    3: "Blur > Beep > Notify > Email > Log",
    4: "Notify > Email > Log",
    5: "Email > Log"
}

# Set Violation Types - Trigger Action Message ALTLANG CHINESE (using ViolationID as key)
# =======================================
violation_types_trigger_action_message_altlang_chinese = {
    1: "僅記錄",
    2: "xx",
    3: "xx > xx > ...",
    4: "xx > xx > ...",
    5: "xx > xx > ..."
}

rule_id_name = {
    1: "Image_Quality",
    2: "Webcam_Align",
    3: "Identity",
    4: "Observers",
    5: "Cell_TakePics",
    6: "Cell_InUse",
    7: "Cell_InArea",
    8: "Keyboards",
    9: "Paper_Books",
    10: "Unattended",
    11: "Distracted",
    12: "Print_Screen"
}

rule_id_name_altlang_chinese = {
    1: "網絡攝像頭被遮擋",
    2: "網絡攝像頭位置",
    3: "越權存取",
    4: "手機",
    5: "手機",
    6: "鍵盤",
    7: "紙",
    8: "缺少用戶",
    9: "打印屏幕",
    10: "無人值守",
    11: "分心",
    12: "打印屏幕"
}

rule_violation_fix_messages_original = {
    1: "Check Webcam Connection",
    2: "Reposition Webcam",
    3: "Look Toward Webcam",
    4: "No Observers Allowed",
    5: "Cell Phones Not Allowed",
    6: "Cell Phones Not Allowed",
    7: "Cell Phones Not Allowed",
    8: "Multiple Keyboards Not Allowed",
    9: "Paper-Notepads Not Allowed",
    10: "Logout When Away",
    11: "Look Toward Webcam",
    12: "Print Screen Not Allowed"
}

rule_violation_fix_messages = {
    1: "Check Webcam Connection",
    2: "Reposition Webcam",
    3: "Look Toward Webcam",
    4: "No Observers Allowed",
    5: "Cell Phones Not Allowed",
    6: "Cell Phones Not Allowed",
    7: "Cell Phones Not Allowed",
    8: "Multiple Keyboards Not Allowed",
    9: "Paper-Notepads Not Allowed",
    10: "Logout When Away",
    11: "Look Toward Webcam",
    12: "Print Screen Not Allowed"
}

rule_violation_fix_messages_altlang_chinese = {
    1: "網絡攝像頭被遮擋",
    2: "網絡攝像頭位置",
    3: "越權存取",
    4: "手機",
    5: "手機",
    6: "鍵盤",
    7: "紙",
    8: "缺少用戶",
    9: "打印屏幕",
    10: "離開時註銷",
    11: "看向網絡攝像頭",
    12: "不允許打印屏幕"
}
