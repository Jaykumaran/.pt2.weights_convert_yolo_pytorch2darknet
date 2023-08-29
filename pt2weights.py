import torch
from cfg import save_conv, save_conv_bn


def save_darknet_weights(model, filename):
    fp = open(filename, 'wb')
    header = torch.IntTensor([0, 2, 0, 0])  # Darknet header
    header.numpy().tofile(fp)

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            if hasattr(module, 'bn'):  # Check if the module has a BatchNormalization layer
                save_conv_bn(fp, module, module.bn)  # Pass both conv_model and bn_model
            else:
                save_conv(fp, module)
        # Add more conditions for other layers if needed

    fp.close()


# Load your YOLOv8 model from the .pt checkpoint file
checkpoint_path = 'C:/Users/jaikr/Downloads/social-distancing-detector-master/social-distancing-detector-master/yolo-coco/yolov8m.pt'
checkpoint = torch.load(checkpoint_path)
yolov8_model = checkpoint["model"]  # Extract the model from the dictionary

# Specify the filename for saving the weights
weights_filename = 'yolo-coco/yolov8n.weights'

# Save the YOLOv8 model weights
save_darknet_weights(yolov8_model, weights_filename)

print(f'Saved YOLOv8 model weights to {weights_filename}')
