
import torch
import torchvision
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader

from dataset import MOT16Dataset, get_transform

def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Data
    dataset = MOT16Dataset(data_dir='data/MOT16/train', annotations_file='data/mot16_annotations.pkl', transform=get_transform(train=True))
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)

    # Model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 2  # 1 class (pedestrian) + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Freeze backbone layers
    for param in model.backbone.parameters():
        param.requires_grad = False

    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Training
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        lr_scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {losses.item()}")

    torch.save(model.state_dict(), 'fasterrcnn_resnet50_fpn_mot16.pth')

if __name__ == '__main__':
    main()
