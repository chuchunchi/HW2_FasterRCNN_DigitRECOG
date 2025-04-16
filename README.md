# HW2_FasterRCNN_DigitRECOG
Homework 2 of VRDL class, digit recognition using Faster R-CNN architecture

Student ID: 313551057

### To run the project from scratch

1. Install python 3.11 and requirements

```
cd HW2_FasterRCNN_DigitRECOG
pip install -r requirements.txt
```

2. download the dataset and store in `\data` folder

3. run training mode

```
python3 src/main.py --mode train
```

### To evaluate the validation data

2. Load the checkpoint file

3. run evaluation or inference

```
python3 python3 src/main.py --mode eval --checkpoint best_model.pth
python3 python3 src/main.py --mode inference --checkpoint best_model.pth
```

### File Architecture
```
digit-recognition/
│
├── data/                       # Data directory
│   ├── train/                  # Training images
│   ├── valid/                  # Validation images
│   ├── test/                   # Test images
│   ├── train.json              # Training annotations (COCO format)
│   └── valid.json              # Validation annotations (COCO format)
│
├── src/
│   ├── main.py                 # Main entry point with configuration, dataset, and utilities
│   ├── models.py               # Model architecture definition (Faster R-CNN)
│   ├── train.py                # Training functionality
│   └── evaluate.py             # Evaluation and inference code
│
├── outputs/                    # Output directory for model checkpoints and results
│   ├── best_model.pth          # Best model checkpoint
│   ├── latest_checkpoint.pth   # Latest model checkpoint
│   ├── checkpoint_epoch_X.pth  # Epoch-specific checkpoints
│   ├── pred.json               # Task 1 predictions (COCO format)
│   ├── pred.csv                # Task 2 predictions
│   ├── val_det.json            # Validation set detections
│   └── task2_results.json      # Task 2 evaluation results
```
