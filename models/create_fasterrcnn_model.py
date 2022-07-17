from models import *


def return_fasterrcnn_resnet50_fpn(
    num_classes, pretrained=True, coco_model=False
):
    model = fasterrcnn_resnet50_fpn.create_model(
        num_classes, pretrained=pretrained, coco_model=coco_model
    )
    return model


def return_fasterrcnn_resnet50(
    num_classes, pretrained=True, coco_model=False
):
    model = fasterrcnn_resnet50.create_model(
        num_classes, pretrained, coco_model
    )
    return model
