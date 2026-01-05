## Dog Detection & Training Assistant

A computer vision and AI-powered application that detects, segments, and classifies dog breeds from images, with integrated LLM chat assistance for dog training guidance.

### Key Features

* Dog Detection & Segmentation: Uses COCO Object Detector to identify and isolate dogs in images
* Breed Classification: Classifies detected dogs using a trained Keras model
* Mask Processing: Applies segmentation masks for improved classification accuracy
* Multi-LLM Support:
    * Ollama (local LLM)
    * Google Gemini (cloud-based)
    * Image generation capabilities
* Dog Training Assistant: Provides personalized advice and guidance based on detected breed
* Confidence Comparison: Compares predictions from regular and masked images to find the most accurate result

## Setup

### Prerequisites

- Python 3.12 or higher
- [UV](https://docs.astral.sh/uv/) - Fast Python package installer

### Installation

1. **Install dependencies using UV**:
   ```bash
   uv sync
   ```

2. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```bash
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

### Verify Installation

```bash
uv run cmd/chat.py ollama
```

## Training Datasets

Stanford Dogs Dataset - http://vision.stanford.edu/aditya86/ImageNetDogs/ for prototype

Sintetic Jack Russell images - https://drive.google.com/drive/folders/19RyDfI06-jzE3wRfoStzPiyWWxhMaMaj?usp=drive_link

## Models training results

| Rank | Model | Train | Val | Gap | Loss (T/V) | Status |
|------|-------|-------|-----|-----|------------|--------|
| 🥇 | **InceptionResNetV2** | 90.5% | **92.5%** | 2% | 0.37/0.35 | ✅ **CHAMPION** |
| 🥈 | **InceptionV3** | 91.5% | **91%** | 0.5% | 0.40/0.47 | ✅ Excellent |
| 🥉 | **ResNet50** | 83% | **80%** | **3%** | 1.05/1.05 | ✅ **Best ResNet50** |

## Usage

### Commands

#### Classify

Segmentation + classification
```
uv run cmd/classify.py <model_path> <image_path> <dataset_path> <result_dir>
```

<small>Example</small>
```
uv run cmd/classify.py storage/models/stanford_dogs_inception_resnet_v2.keras storage/test-images/jrt-1.jpg storage/datasets/stanford-dogs-dataset storage/ts
```

#### Predict
Classification
```
uv run cmd/predict.py <model_path> <image_path> <dataset_path>
```

<small>Example</small>
```
uv run cmd/predict.py storage/models/stanford_dogs_inception_resnet_v2.keras storage/test-images/jrt-1.jpg storage/datasets/stanford-dogs-dataset
```

#### coco_detect
Segmentation
```
uv run cmd/coco_detect.py <image_path> <result_dir>
```

<small>Example</small>
```
uv run cmd/coco_detect.py storage/test-images/jrt-1.jpg storage/ts
```

#### train
```
uv run cmd/train.py <cnn_type> <dataset_path> --model_name=<name>
```

<small>Example</small>
```
uv run cmd/train.py inception_resnet_v2 ./storage/datasets/stanford-dogs-dataset --model_name=stanford_dogs_inception_resnet_v2
```

##### CNN types:
1. vgg19
2. resnet50
3. inception_v3
4. custom
5. inception_resnet_v2
6. efficientnet_b7

##### model_name
If missing the model type will be used as name.


#### chat
Chat with AI assistance
```
uv run cmd/chat.py <assistance_type>
```

<small>Example</small>
```
uv run cmd/chat.py ollama
```

##### Assistance types:
1. ollama
2. gemini
