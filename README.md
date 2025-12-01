## Dataset

Stanford Dogs Dataset - http://vision.stanford.edu/aditya86/ImageNetDogs/ for prototype


## Commands

### Classify
Segmentation + classification
```
uv run cmd/classify.py storage/models/{MODEL}.keras storage/test-images/{IMAGE} storage/datasets/stanford-dogs-dataset storage/{RESULT DIR}
```

### Predict
Classification
```
uv run cmd/predict.py storage/models/{MODEL}.keras storage/test-images/{IMAGE} storage/datasets/stanford-dogs-dataset
```

### coco_detect
Segmentation
```
uv run cmd/coco_detect.py storage/test-images/{IMAGE} storage/{RESULT DIR}
```