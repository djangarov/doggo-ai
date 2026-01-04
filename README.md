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

### train
```
uv run cmd/train.py {CNN TYPE} storage/datasets/stanford-dogs-dataset --model_name={MODEL NAME}
```

##### CNN TYPES:
1. vgg19
2. resnet50
3. inception_v3
4. custom
5. inception_resnet_v2
6. efficientnet_b7

##### model_name
If missing the model type will be used as name.


### chat
Chat with AI assistance
```
uv run cmd/chat.py {ASSISTANCE TYPE}
```

##### ASSIStANCE TYPES:
1. ollama
2. gemini
