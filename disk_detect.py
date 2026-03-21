import os
import xml.etree.ElementTree as ET
from ultralytics import YOLO
import cv2
import shutil

class DiscDetector:
    classes = ["disk"]
    xml_dir = "xml"
    train_img_dir = "img/model1_train"
    val_img_dir = "img/model1_val"
    train_label_dir = "labels/model1_train"
    val_label_dir = "labels/model1_val"
    model_path = "best.pt"
    model = None

    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)

    @staticmethod
    def get_model():
        if DiscDetector.model is None:
            DiscDetector.model = YOLO(DiscDetector.model_path)
        return DiscDetector.model

    @staticmethod
    def convert_box(size, box):
        dw = 1.0 / size[0]
        dh = 1.0 / size[1]

        xmin, xmax, ymin, ymax = box
        x_center = ((xmin + xmax) / 2.0) * dw
        y_center = ((ymin + ymax) / 2.0) * dh
        width = (xmax - xmin) * dw
        height = (ymax - ymin) * dh

        return x_center, y_center, width, height

    @staticmethod
    def convert_xml(xml_path, output_txt_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find("size")
        img_w = int(size.find("width").text)
        img_h = int(size.find("height").text)

        lines = []

        for obj in root.findall("object"):
            cls_name = obj.find("name").text.strip()

            if cls_name not in DiscDetector.classes:
                continue

            cls_id = DiscDetector.classes.index(cls_name)
            xmlbox = obj.find("bndbox")

            xmin = float(xmlbox.find("xmin").text)
            xmax = float(xmlbox.find("xmax").text)
            ymin = float(xmlbox.find("ymin").text)
            ymax = float(xmlbox.find("ymax").text)

            x, y, w, h = DiscDetector.convert_box(
                (img_w, img_h),
                (xmin, xmax, ymin, ymax)
            )
            lines.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

        with open(output_txt_path, "w") as f:
            f.write("\n".join(lines))

    @staticmethod
    def process_split(image_dir, label_dir):
        image_basenames = set()

        for fname in os.listdir(image_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                image_basenames.add(os.path.splitext(fname)[0])

        for xml_file in os.listdir(DiscDetector.xml_dir):
            if not xml_file.endswith(".xml"):
                continue

            base = os.path.splitext(xml_file)[0]
            if base in image_basenames:
                xml_path = os.path.join(DiscDetector.xml_dir, xml_file)
                txt_path = os.path.join(label_dir, base + ".txt")
                DiscDetector.convert_xml(xml_path, txt_path)
                print(f"Converted: {xml_file} -> {txt_path}")

    @staticmethod
    def convert_xml_to_txt():
        DiscDetector.process_split(
            DiscDetector.train_img_dir,
            DiscDetector.train_label_dir
        )
        DiscDetector.process_split(
            DiscDetector.val_img_dir,
            DiscDetector.val_label_dir
        )
        print("Done.")

    @staticmethod
    def predict_disk_coords(img_path):
        original_img = cv2.imread(img_path)
        if original_img is None:
            return {}

        folder, filename = os.path.split(img_path)

        # enhanced image is only for detection
        enhanced_img_path = DiscDetector.enhance_img(original_img, folder, filename)

        model = DiscDetector.get_model()
        results = model(enhanced_img_path)

        best_coords = {}
        best_area = -1

        for res in results:
            boxes = res.boxes.xyxy
            for box in boxes:
                xmin, ymin, xmax, ymax = box.tolist()
                area = (xmax - xmin) * (ymax - ymin)

                if area > best_area:
                    best_area = area
                    best_coords = {
                        "xmin": xmin,
                        "xmax": xmax,
                        "ymin": ymin,
                        "ymax": ymax
                    }

        return best_coords

    @staticmethod
    def crop_image(img_path, coords):
        image = cv2.imread(img_path)
        if image is None:
            return None

        h, w = image.shape[:2]

        xmin = max(0, int(coords["xmin"]))
        xmax = min(w, int(coords["xmax"]))
        ymin = max(0, int(coords["ymin"]))
        ymax = min(h, int(coords["ymax"]))

        if xmin >= xmax or ymin >= ymax:
            return None

        img_crop = image[ymin:  ymax, xmin:xmax]
        return img_crop

    @staticmethod
    def enhance_img(img_obj, folder, filename):
        out_folder = os.path.join(folder, "cleaned")
        os.makedirs(out_folder, exist_ok=True)

        name, ext = os.path.splitext(filename)
        cleaned_name = name + "_cleaned" + ext
        out_path = os.path.join(out_folder, cleaned_name)

        green = img_obj[:, :, 1]

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        green_clahe = clahe.apply(green)
        blurred_img = cv2.GaussianBlur(green_clahe, (5, 5), 0)

        cv2.imwrite(out_path, blurred_img)
        return out_path

    @staticmethod
    def preprocess(list_img, output_dir):
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir) 

        os.makedirs(output_dir, exist_ok=True)

        for img_path in list_img:
            if not os.path.isfile(img_path):
                print(f"Skipped (file not found): {img_path}")
                continue

            filename = os.path.basename(img_path)

            coords = DiscDetector.predict_disk_coords(img_path)
            if not coords:
                print(f"Skipped (no detection): {filename}")
                continue

            cropped_img = DiscDetector.crop_image(img_path, coords)
            if cropped_img is None or cropped_img.size == 0:
                print(f"Skipped (invalid crop): {filename}")
                continue

            output_path = os.path.join(output_dir, filename)

            success = cv2.imwrite(output_path, cropped_img)
            if success:
                print(f"Saved: {output_path}")
            else:
                print(f"Skipped (save failed): {filename}")


if __name__ == "__main__":
    
    #GET IMAGE PATH
    #CALL PREDICT, RETURN : COORDS DICT
    """ 
    coords = {
        "xmin", "xmax", "ymin","ymax"
    } 
    """

    #CALL CROP 

    #TO DO
    #FIT IN INTO CLASSIFIER



    pass