# Copyright 2018 by Peter Cock, The James Hutton Institute.
# All rights reserved.

# Based on https://medium.com/@hungminhnguyen/convert-fast-ai-trained-image-classification-model-to-ios-app-via-onnx-and-apple-core-ml-5fdb612379f1

from fastai.layers import AdaptiveConcatPool2d

import torch
from torch.autograd import Variable
from torch import nn
import torch.onnx
import onnx

class MonkyPatchedAdaptiveMaxPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        # why (10, 10)? Because input image size is 299, \
        # if you use 224, this should be (7, 7)
        # if you want to know which number for other image size,
        # put pdb.set_trace() at forward method and print x.size()
        self.p = nn.MaxPool2d((10, 10), padding=0)

    def forward(self, x):
        return self.p(x)

class MonkyPatchedAdaptiveAvgPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        self.p = nn.AvgPool2d((10, 10), padding=0)

    def forward(self, x):
        return self.p(x)

def MonkyPatched_AdaptiveConcatPool2d_init(self, sz=None):
        super(AdaptiveConcatPool2d, self).__init__()
        sz = sz or (1,1)
        self.ap = MonkyPatchedAdaptiveAvgPool2d(sz)
        self.mp = MonkyPatchedAdaptiveMaxPool2d(sz)

def monkeypatch_fastai_for_onnx():
    AdaptiveConcatPool2d.__init__ = MonkyPatched_AdaptiveConcatPool2d_init

def convert_and_validate_to_onnx(model, model_name, sz, input_names, output_names):
    dummy_input = Variable(torch.randn(3, sz, sz)).cuda()

    torch.onnx.export(model, dummy_input, \
                      model_name, input_names=input_names,
                      output_names=output_names, verbose=True)

    # Check again by onnx
    # Load the ONNX model
    onnx_model = onnx.load(model_name)

    # Check that the IR is well formed
    onnx.checker.check_model(onnx_model)

    # Print a human readable representation of the graph
    #     onnx.helper.printable_graph(onnx_model.graph)

    return onnx_model


class ImageScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.denorminator = torch.full((3, 299, 299), 255.0, device=torch.device("cuda"))

    def forward(self, x): return torch.div(x, self.denorminator).unsqueeze(0)

def make_fastai_be_coreml_compatible(learner):
    # Suggestion for using softmax didn't work for me
    #final_model = [ImageScale()] + (list(learn.model.children())[:-1] + [nn.Softmax()])
    return nn.Sequential(ImageScale(), *learner.model)

if __name__== "__main__":
    foo = AdaptiveConcatPool2d()
    monkeypatch_fastai_for_onnx()
    bar = AdaptiveConcatPool2d()