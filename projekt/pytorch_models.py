import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
from PIL import Image
import os
from functools import partial
import pynvml
import psutil
import threading
import datetime
import functions

number_of_epochs = 10

type_of_data = "objects"

COCO_CLASSES = {0: 'background', 1: 'person'}
used_models = ["fasterrcnn", "retinanet", "maskrcnn"]

model_dir_path = "./models/"
if not os.path.exists(model_dir_path):
    os.makedirs(model_dir_path)

#transform data to COCO format
class CocoTransform:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

def prepare_image(image_path, device):
    """
    * Load image in RGB
    * Transform to tensor
    * Move image to device

    Input:
        - image_path:
            - path to image
        - device
            - device instance
    Output:
        - Moved image to device
    """
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0)
    return image_tensor.to(device)

def get_class_name(class_id):
    """
    Getter for class name

    Input:
        - class_id;
            - id of object class
    Output:
        - name of object class
    """
    return COCO_CLASSES.get(class_id, "Unknown")

def get_coco_dataset(img_dir, annotation_file):
    """
    Load dataset in COCO format

    Input:
        - img_dir:
            - path to test image directory
        - annotation_file
            - test annotations for imagess
    Output:
        - COCO detection dataset
    """
    return CocoDetection(
        root=img_dir,
        annFile=annotation_file,
        transforms=CocoTransform()
    )

def get_model(model, num_classes):
    """
    Get model instance

    Input:
        - model:
            - text represetation of model
        - num_classes:
            - count of classes
    Output:
        - model instance
    """
    if model == "fasterrcnn":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    elif model == "retinanet":
        model = torchvision.models.detection.retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT)
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = RetinaNetClassificationHead(
            in_channels=256,
            num_anchors=num_anchors,
            num_classes=num_classes,
            norm_layer=partial(torch.nn.GroupNorm, 32)
        )
    elif model == "maskrcnn":
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    
    return model

def train_one_epoch(model, optimizer, data_loader, device, epoch, add_mask):
    """
    Train model for one epoch

    Input:
        - model:
            - model instance
        - optimizer:
            - optimizer instance
        - data_loader:
            - train image dataset
        - device:
            - device instance
        - epoch:
            - count of trainning epochs
        - add_mask:
            - boolean mask switch
    """
    model.train()
    for images, targets in data_loader:
        images = [image.to(device) for image in images]

        processed_targets = []
        valid_images = []
        for index, target in enumerate(targets):
            boxes = []
            labels = []
            image = images[index]
            for obj in target:
                bbox = obj['bbox']
                x, y, w, h = bbox

                if w > 0 and h > 0:
                    boxes.append([x, y, x + w, y + h])
                    labels.append(obj['category_id'])
            
            if boxes:
                boxes = torch.tensor(boxes, dtype=torch.float32).to(device)
                labels = torch.tensor(labels, dtype=torch.int64).to(device)
                if add_mask:
                    img_height, img_width = image.shape[1:]
                    masks = torch.zeros((len(labels), img_height, img_width), dtype=torch.bool).to(device)
                    processed_targets.append({'boxes': boxes, 'labels': labels, 'masks': masks})
                else:
                    processed_targets.append({'boxes': boxes, 'labels': labels})
                valid_images.append(image)
        
        if not processed_targets:
            continue

        images = valid_images
        loss_dict = model(images, processed_targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f"Epoch [{epoch}] loss:{losses.item():.4f}.")

def train_model(model_text, train_dataloader, number_of_epochs):
    """
    Train model

    Input:
        - model_text:
            - text represetation of model
        - train_dataloader:
            - train loaded data
        - number_of_epochs:
            - count of epochs
    """
    num_classes = 2
    model = get_model(model_text, num_classes)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(number_of_epochs):
        if model_text == "maskrcnn":
            train_one_epoch(model, optimizer, train_dataloader, device, epoch, True)
        else:
            train_one_epoch(model, optimizer, train_dataloader, device, epoch, False)
        lr_scheduler.step()

        if epoch == number_of_epochs - 1:
            model_path = os.path.join(model_dir_path, f"{model_text}_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), model_path)

def load_image_paths(img_dir_path):
    """
    Load image paths

    Input:
        - img_dir_path:
            - path to directory with images
    Output:
        - array of images paths
    """
    array_of_paths = []
    array_of_images = os.listdir(img_dir_path)
    for image in array_of_images:
        array_of_paths.append(os.path.join(img_dir_path, image))
    
    return array_of_paths

def eval_img_model(model_text, array_of_image_paths, number_of_epochs):
    """
    * Test model
    * Measure the time of run between request and response of model is seconds.
    * Measure CPU/GPU and RAM/VRAM usage

    Input:
        - model_text
            - text representation of model
        - array_of_image_paths
            - array of test images paths
        - number_of_epochs
            - number of trained epochs
    """
    num_classes = 2
    model = get_model(model_text, num_classes)
    model.load_state_dict(torch.load(os.path.join(model_dir_path, f"{model_text}_epoch_{number_of_epochs}.pth")))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    model.eval()
    with torch.no_grad():
        for image in array_of_image_paths:
            #get process id
            pid = os.getpid()
            process = psutil.Process(pid)

            #GPU init
            gpu_handle = None
            base_vram_mb = 0.0
            try:
                pynvml.nvmlInit()
                gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
                base_vram_mb = info.used / (1024 * 1024)
                gpu_is_available = True
            except pynvml.NVMLError:
                print("NVIDIA GPU not found.")
                gpu_is_available = False
            
            detections = []
            
            #init of thread
            functions.monitor_data["is_running"] = True
            monitor_thread = threading.Thread(
            target=functions.monitor_memory_gpu_vram, 
            args=(process, gpu_handle),
            daemon=True #stops if main script stops
            )
            monitor_thread.start()
            vram_after = 0.0

            #cpu and memory before test model
            process.cpu_percent(interval=None)
            mem_before = process.memory_info().rss / (1024 * 1024)

            start_datetime = datetime.datetime.now()

            try:
                image_tensor = prepare_image(image, device)
                prediction = model(image_tensor)

                boxes = prediction[0]['boxes'].cpu().numpy()
                labels = prediction[0]['labels'].cpu().numpy()
                scores = prediction[0]['scores'].cpu().numpy()

                for box, label, score in zip(boxes, labels, scores):
                    x_min, y_min, x_max, y_max = box
                    class_name = get_class_name(label)

                    detections.append({
                        "class_name": class_name,
                        "x_min": x_min,
                        "y_min": y_min,
                        "x_max": x_max,
                        "y_max": y_max,
                        "confidence" : score
                    })
            finally:
                # stop thread
                functions.monitor_data["is_running"] = False
                monitor_thread.join(timeout=1.0)
                if gpu_is_available:
                    vram_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
                    vram_after = vram_info.used / (1024 * 1024)
                    pynvml.nvmlShutdown() #shutdown nvml
            
            end_datetime = datetime.datetime.now()
            #get cpu and ram usage
            mem_after = process.memory_info().rss / (1024 * 1024)
            peak_ram_mb = functions.monitor_data["peak_rss_mb"]
            cpu_usage = process.cpu_percent(interval=None)
    
            peak_ram_mb = max(peak_ram_mb, mem_after) #maximum of peak RAM and final value of RAM
            ram_usage = peak_ram_mb - mem_before

            #GPU VRAM usage
            if gpu_is_available:
                total_vram_mb = max(functions.monitor_data["peak_vram_mb"], vram_after) - base_vram_mb
            else:
                total_vram_mb = -1
            
            #time of test
            diff_datetime = end_datetime - start_datetime
            diff_datetime_seconds = diff_datetime.total_seconds()

            file_name = os.path.basename(image)

            max_iou_detections, good_boxes = functions.get_max_iou_and_good_boxes(file_name, detections)

            for iou_threshold in functions.iou_thresholds:
                map_values = functions.get_mAP(max_iou_detections, good_boxes, iou_threshold)
                functions.save_to_file_object(model_text, type_of_data, map_values["map"],
                                              map_values["map_50"], map_values["map_75"],
                                              map_values["map_large"], map_values["mar_100"],
                                              map_values["mar_large"], iou_threshold)
            functions.save_to_file_object_main(model_text, type_of_data, diff_datetime_seconds, 0)

            functions.save_to_file_cpu_gpu(model_text, type_of_data, True, cpu_usage, functions.monitor_data["peak_cpu_percent"],
                                           ram_usage, functions.monitor_data["peak_gpu_utilization"], total_vram_mb,
                                           0) #this information is in other file there

def load_and_measure(model, train_dataloader, train_switch, eval_switch):
    """
    * Measure the time of run between request and response of model is seconds.
    * Measure CPU/GPU and RAM/VRAM usage
    * Train model
    * Call test model

    Input:
        - model
            - text representation of model
        - train_dataloader
            - loaded train data
        - train_switch
            - switch for train model
        - eval_switch
            - switch for test model
    """
    if train_switch:
        #get process id
        pid = os.getpid()
        process = psutil.Process(pid)
    
        #GPU init
        gpu_handle = None
        base_vram_mb = 0.0
        try:
            pynvml.nvmlInit()
            gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
            base_vram_mb = info.used / (1024 * 1024)
            gpu_is_available = True
        except pynvml.NVMLError:
            print("NVIDIA GPU not found.")
            gpu_is_available = False

        functions.monitor_data["is_running"] = True
        monitor_thread = threading.Thread(
            target=functions.monitor_memory_gpu_vram, 
            args=(process, gpu_handle),
            daemon=True #stops if main script stops
        )
        monitor_thread.start()
        vram_after = 0.0

        #cpu and memory before test model
        process.cpu_percent(interval=None)
        mem_before = process.memory_info().rss / (1024 * 1024)
        start_datetime = datetime.datetime.now()

        try:
            train_model(model, train_dataloader, number_of_epochs)
        finally:
            # stop thread
            functions.monitor_data["is_running"] = False
            monitor_thread.join(timeout=1.0)
            if gpu_is_available:
                vram_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
                vram_after = vram_info.used / (1024 * 1024)
                pynvml.nvmlShutdown() #shutdown nvml
        
        end_datetime = datetime.datetime.now()
        #get cpu and ram usage
        mem_after = process.memory_info().rss / (1024 * 1024)
        peak_ram_mb = functions.monitor_data["peak_rss_mb"]
        cpu_usage = process.cpu_percent(interval=None)

        peak_ram_mb = max(peak_ram_mb, mem_after) #maximum of peak RAM and final value of RAM
        ram_usage = peak_ram_mb - mem_before

        #GPU VRAM usage
        if gpu_is_available:
            total_vram_mb = max(functions.monitor_data["peak_vram_mb"], vram_after) - base_vram_mb
        else:
            total_vram_mb = -1

        #time of train
        diff_datetime = end_datetime - start_datetime
        diff_datetime_seconds = diff_datetime.total_seconds()

        functions.save_to_file_cpu_gpu(model, type_of_data, False, cpu_usage, functions.monitor_data["peak_cpu_percent"],
                                       ram_usage, functions.monitor_data["peak_gpu_utilization"], total_vram_mb, diff_datetime_seconds)
    if eval_switch:
        eval_img_model(model, load_image_paths("../dataset/yolo_dataset/test/images"), number_of_epochs)

if __name__ == "__main__":
    train_dataset = get_coco_dataset("../dataset/yolo_dataset/train/images", "../dataset/coco_annotations/train/annotations.json")
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    load_and_measure("fasterrcnn", train_dataloader, False, True)
    load_and_measure("retinanet", train_dataloader, False, True)
    load_and_measure("maskrcnn", train_dataloader, False, True)