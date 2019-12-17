# fastai_to_coreml

Helpers to convert Fast.ai models to CoreML.  Based on https://medium.com/@hungminhnguyen/convert-fast-ai-trained-image-classification-model-to-ios-app-via-onnx-and-apple-core-ml-5fdb612379f1


## Usage

Before training, call 

```
import fastai_to_coreml
fastai_to_coreml.monkeypatch_fastai_for_onnx()
```

After training, writing labels to a file:

```
with open('labels.txt', 'w') as f:
    for item in data.classes:
        f.write("%s\n" % item)
  ```    
Then use onnx_coreml to convert

```
from onnx_coreml import convert

# Convert Pytorch model to onnx model & check if it is convertible
onnx_model = fastai_to_coreml.convert_and_validate_to_onnx(
    fastai_to_coreml.make_fastai_be_coreml_compatible(learn), 
    model_name, 299, input_names=['image'], output_names=['pose'])

# Convert onnx model to Apple Core ML
mlmodel = convert(onnx.load(model_name), image_input_names = ['image'], mode='classifier', class_labels="labels.txt")
...
mlmodel.save(f'{model_name}.mlmodel')
```

