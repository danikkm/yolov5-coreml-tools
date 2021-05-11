# Copyright (C) 2021 DB Systel GmbH.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import coremltools as ct
from argparse import ArgumentParser
from pathlib import Path

# Add silu function for yolov5s v4 model: https://github.com/apple/coremltools/issues/1099
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil import register_torch_op
from coremltools.converters.mil.frontend.torch.ops import _get_inputs

import yaml


@register_torch_op
def silu(context, node):
  inputs = _get_inputs(context, node, expected=1)
  x = inputs[0]
  y = mb.sigmoid(x=x)
  z = mb.mul(x=x, y=y, name=node.name)
  context.add(z)


def read_coco_dataset_labels(opt):
  dataset_file = {}
  labels = []
  with open(opt.coco_dataset_file, 'r') as file:
    dataset_file = yaml.load(file, Loader=yaml.FullLoader)

  try:
    labels = dataset_file['names']
  except Exception as e:
    assert False, 'Error: invalid coco dataset file provided'

  return labels


def make_grid(nx, ny):
  yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
  return torch.stack((xv, yv), 2).view((ny, nx, 2)).float()


def export_torchscript(model, sample_input, check_inputs, file_name):
  '''
  Traces a pytorch model and produces a TorchScript
  '''
  try:
    print(f'Starting TorchScript export with torch {torch.__version__}')
    ts = torch.jit.trace(model, sample_input, check_inputs=check_inputs)
    ts.save(file_name)
    print(f'TorchScript export success, saved as {file_name}')
    return ts
  except Exception as e:
    print(f'TorchScript export failure: {e}')


def convert_to_coreml_spec(torch_script, sample_input):
  '''
  Converts a torchscript to a coreml model
  '''
  try:
    print(f'Starting CoreML conversion with coremltools {ct.__version__}')
    nn_spec = ct.convert(torch_script, inputs=[ct.ImageType(
        name='image', shape=sample_input.shape, scale=1 / 255.0, bias=[0, 0, 0])]).get_spec()

    print(f'CoreML conversion success')
  except Exception as e:
    print(f'CoreML conversion failure: {e}')
    return
  return nn_spec


def add_output_meta_data(nn_spec, feature_map_dimensions, output_size):
  '''
  Adds the correct output shapes and data types to the coreml model
  '''
  for i, feature_map_dimension in enumerate(feature_map_dimensions):
    nn_spec.description.output[i].type.multiArrayType.shape.append(1)
    nn_spec.description.output[i].type.multiArrayType.shape.append(3)
    nn_spec.description.output[i].type.multiArrayType.shape.append(
        feature_map_dimension)
    nn_spec.description.output[i].type.multiArrayType.shape.append(
        feature_map_dimension)
    # pc, bx, by, bh, bw, c (no of class class labels)
    nn_spec.description.output[i].type.multiArrayType.shape.append(
        output_size)
    nn_spec.description.output[i].type.multiArrayType.dataType = ct.proto.FeatureTypes_pb2.ArrayFeatureType.DOUBLE


def add_export_layer_to_coreml(opt, builder, anchor_grid, feature_map_dimensions, strides, number_of_class_labels):
  '''
  Adds the yolov5 export layer to the coreml model
  '''
  output_names = [output.name for output in builder.spec.description.output]

  for i, output_name in enumerate(output_names):
    # formulas: https://github.com/ultralytics/yolov5/issues/471
    builder.add_activation(name=f'sigmoid_{output_name}', non_linearity='SIGMOID',
                           input_name=output_name, output_name=f'{output_name}_sigmoid')

    ### Coordinates calculation ###
    # input (1, 3, nC, nC, 85), output (1, 3, nC, nC, 2) -> nC = 640 / strides[i]
    builder.add_slice(name=f'slice_coordinates_xy_{output_name}', input_name=f'{output_name}_sigmoid',
                      output_name=f'{output_name}_sliced_coordinates_xy', axis='width', start_index=0, end_index=2)
    # x,y * 2
    builder.add_elementwise(name=f'multiply_xy_by_two_{output_name}', input_names=[
        f'{output_name}_sliced_coordinates_xy'], output_name=f'{output_name}_multiplied_xy_by_two', mode='MULTIPLY', alpha=2)
    # x,y * 2 - 0.5
    builder.add_elementwise(name=f'subtract_0_5_from_xy_{output_name}', input_names=[
        f'{output_name}_multiplied_xy_by_two'], output_name=f'{output_name}_subtracted_0_5_from_xy', mode='ADD', alpha=-0.5)
    grid = make_grid(
        feature_map_dimensions[i], feature_map_dimensions[i]).numpy()
    # x,y * 2 - 0.5 + grid[i]
    builder.add_bias(name=f'add_grid_from_xy_{output_name}', input_name=f'{output_name}_subtracted_0_5_from_xy',
                     output_name=f'{output_name}_added_grid_xy', b=grid, shape_bias=grid.shape)
    # (x,y * 2 - 0.5 + grid[i]) * stride[i]
    builder.add_elementwise(name=f'multiply_xy_by_stride_{output_name}', input_names=[
        f'{output_name}_added_grid_xy'], output_name=f'{output_name}_calculated_xy', mode='MULTIPLY', alpha=strides[i])

    # input (1, 3, nC, nC, 85), output (1, 3, nC, nC, 2)
    builder.add_slice(name=f'slice_coordinates_wh_{output_name}', input_name=f'{output_name}_sigmoid',
                      output_name=f'{output_name}_sliced_coordinates_wh', axis='width', start_index=2, end_index=4)
    # w,h * 2
    builder.add_elementwise(name=f'multiply_wh_by_two_{output_name}', input_names=[
        f'{output_name}_sliced_coordinates_wh'], output_name=f'{output_name}_multiplied_wh_by_two', mode='MULTIPLY', alpha=2)
    # (w,h * 2) ** 2
    builder.add_unary(name=f'power_wh_{output_name}', input_name=f'{output_name}_multiplied_wh_by_two',
                      output_name=f'{output_name}_power_wh', mode='power', alpha=2)
    # (w,h * 2) ** 2 * anchor_grid[i]
    anchor = anchor_grid[i].expand(-1, feature_map_dimensions[i],
                                   feature_map_dimensions[i], -1).numpy()
    builder.add_load_constant_nd(
        name=f'anchors_{output_name}', output_name=f'{output_name}_anchors', constant_value=anchor, shape=anchor.shape)
    builder.add_elementwise(name=f'multiply_wh_with_achors_{output_name}', input_names=[
        f'{output_name}_power_wh', f'{output_name}_anchors'], output_name=f'{output_name}_calculated_wh', mode='MULTIPLY')

    builder.add_concat_nd(name=f'concat_coordinates_{output_name}', input_names=[
        f'{output_name}_calculated_xy', f'{output_name}_calculated_wh'], output_name=f'{output_name}_raw_coordinates', axis=-1)
    builder.add_scale(name=f'normalize_coordinates_{output_name}', input_name=f'{output_name}_raw_coordinates',
                      output_name=f'{output_name}_raw_normalized_coordinates', W=torch.tensor([1 / opt.img_size]).numpy(), b=0, has_bias=False)

    ### Confidence calculation ###
    builder.add_slice(name=f'slice_object_confidence_{output_name}', input_name=f'{output_name}_sigmoid',
                      output_name=f'{output_name}_object_confidence', axis='width', start_index=4, end_index=5)
    builder.add_slice(name=f'slice_label_confidence_{output_name}', input_name=f'{output_name}_sigmoid',
                      output_name=f'{output_name}_label_confidence', axis='width', start_index=5, end_index=0)
    # confidence = object_confidence * label_confidence
    builder.add_multiply_broadcastable(name=f'multiply_object_label_confidence_{output_name}', input_names=[
        f'{output_name}_label_confidence', f'{output_name}_object_confidence'], output_name=f'{output_name}_raw_confidence')

    # input: (1, 3, nC, nC, 85), output: (3 * nc^2, 85)
    builder.add_flatten_to_2d(
        name=f'flatten_confidence_{output_name}', input_name=f'{output_name}_raw_confidence', output_name=f'{output_name}_flatten_raw_confidence', axis=-1)
    builder.add_flatten_to_2d(
        name=f'flatten_coordinates_{output_name}', input_name=f'{output_name}_raw_normalized_coordinates', output_name=f'{output_name}_flatten_raw_coordinates', axis=-1)

  builder.add_concat_nd(name='concat_confidence', input_names=[
      f'{output_name}_flatten_raw_confidence' for output_name in output_names], output_name='raw_confidence', axis=-2)
  builder.add_concat_nd(name='concat_coordinates', input_names=[
      f'{output_name}_flatten_raw_coordinates' for output_name in output_names], output_name='raw_coordinates', axis=-2)

  builder.set_output(output_names=['raw_confidence', 'raw_coordinates'], output_dims=[
      (3 * ((opt.img_size // 8)**2 + (opt.img_size // 16)**2 + (opt.img_size // 32)**2),
       number_of_class_labels), (3 * ((opt.img_size // 8)**2 + (opt.img_size // 16)**2 + (opt.img_size // 32)**2), 4)])


def create_nms_model_spec(nn_spec, number_of_class_labels, class_labels):
  '''
  Create a coreml model with nms to filter the results of the model
  '''
  nms_spec = ct.proto.Model_pb2.Model()
  nms_spec.specificationVersion = 4

  # Define input and outputs of the model
  for i in range(2):
    nnOutput = nn_spec.description.output[i].SerializeToString()

    nms_spec.description.input.add()
    nms_spec.description.input[i].ParseFromString(nnOutput)

    nms_spec.description.output.add()
    nms_spec.description.output[i].ParseFromString(nnOutput)

  nms_spec.description.output[0].name = 'confidence'
  nms_spec.description.output[1].name = 'coordinates'

  # Define output shape of the model
  output_sizes = [number_of_class_labels, 4]
  for i in range(len(output_sizes)):
    maType = nms_spec.description.output[i].type.multiArrayType
    # First dimension of both output is the number of boxes, which should be flexible
    maType.shapeRange.sizeRanges.add()
    maType.shapeRange.sizeRanges[0].lowerBound = 0
    maType.shapeRange.sizeRanges[0].upperBound = -1
    # Second dimension is fixed, for 'confidence' it's the number of classes, for coordinates it's position (x, y) and size (w, h)
    maType.shapeRange.sizeRanges.add()
    maType.shapeRange.sizeRanges[1].lowerBound = output_sizes[i]
    maType.shapeRange.sizeRanges[1].upperBound = output_sizes[i]
    del maType.shape[:]

  # Define the model type non maximum supression
  nms = nms_spec.nonMaximumSuppression
  nms.confidenceInputFeatureName = 'raw_confidence'
  nms.coordinatesInputFeatureName = 'raw_coordinates'
  nms.confidenceOutputFeatureName = 'confidence'
  nms.coordinatesOutputFeatureName = 'coordinates'
  nms.iouThresholdInputFeatureName = 'iouThreshold'
  nms.confidenceThresholdInputFeatureName = 'confidenceThreshold'
  # Some good default values for the two additional inputs, can be overwritten when using the model
  nms.iouThreshold = 0.6
  nms.confidenceThreshold = 0.4

  nms.stringClassLabels.vector.extend(class_labels)

  return nms_spec


def combine_models_and_export(opt, builder_spec, nms_spec, file_name, quantize=False):
  '''
  Combines the coreml model with export logic and the nms to one final model. Optionally save with different quantization (32, 16, 8) (Works only if on Mac Os)
  '''
  try:
    print(f'Combine CoreMl model with nms and export model')
    # Combine models to a single one
    pipeline = ct.models.pipeline.Pipeline(input_features=[('image', ct.models.datatypes.Array(3, opt.img_size, opt.img_size)),
                                                           ('iouThreshold', ct.models.datatypes.Double(
                                                           )),
                                                           ('confidenceThreshold', ct.models.datatypes.Double())], output_features=['confidence', 'coordinates'])

    # Required version (>= ios13) in order for mns to work
    pipeline.spec.specificationVersion = 4

    pipeline.add_model(builder_spec)
    pipeline.add_model(nms_spec)

    pipeline.spec.description.input[0].ParseFromString(
        builder_spec.description.input[0].SerializeToString())
    pipeline.spec.description.output[0].ParseFromString(
        nms_spec.description.output[0].SerializeToString())
    pipeline.spec.description.output[1].ParseFromString(
        nms_spec.description.output[1].SerializeToString())

    # Metadata for the model‚
    pipeline.spec.description.input[0].shortDescription = f'{opt.img_size}x{opt.img_size} RGB Image'
    pipeline.spec.description.input[1].shortDescription = '(optional) IOU Threshold override (Default: 0.6)'
    pipeline.spec.description.input[
        2].shortDescription = '(optional) Confidence Threshold override (Default: 0.4)'
    pipeline.spec.description.output[0].shortDescription = u'Boxes \xd7 Class confidence'
    pipeline.spec.description.output[
        1].shortDescription = u'Boxes \xd7 [x, y, width, height] (relative to image size)'
    pipeline.spec.description.metadata.versionString = '1.0'
    pipeline.spec.description.metadata.shortDescription = opt.model_output_name
    pipeline.spec.description.metadata.author = ''
    pipeline.spec.description.metadata.license = ''

    model = ct.models.MLModel(pipeline.spec)
    model.save(file_name)

    if quantize:
      file_name16 = file_name.replace('.mlmodel', '_16.mlmodel')
      model_fp16 = ct.models.neural_network.quantization_utils.quantize_weights(
          model, nbits=16)
      model_fp16.save(file_name16)

      file_name8 = file_name.replace('.mlmodel', '_8.mlmodel')
      model_fp8 = ct.models.neural_network.quantization_utils.quantize_weights(
          model, nbits=8)
      model_fp8.save(file_name8)

    print(f'CoreML export success, saved as {file_name}')
  except Exception as e:
    print(f'CoreML export failure: {e}')


def main():
  parser = ArgumentParser()
  parser.add_argument('--model-input-path', type=str, dest='model_input_path',
                      default='models/yolov5s_v4.pt', help='path to yolov5 model')
  parser.add_argument('--model-output-directory', type=str,
                      dest='model_output_directory', default='output/models', help='model output path')
  parser.add_argument('--model-output-name', type=str, dest='model_output_name',
                      default='yolov5-iOS', help='model output name')
  parser.add_argument('--img-size', type=int, dest='img_size',
                      default='416', help='size of the output image')
  parser.add_argument('--coco-dataset-file', type=str, dest='coco_dataset_file',
                      help='path to the coco dataset file, e.g., coco.yaml')
  parser.add_argument('--quantize-model', action='store_true', dest='quantize',
                      help='Pass flag quantized models are needed (Only works on macOS)')
  opt = parser.parse_args()

  if not Path(opt.model_input_path).exists():
    assert False, 'Error: Input model not found'

  if not opt.coco_dataset_file:
    assert False, 'Error: Please provide path to the coco dataset file'

  class_labels = read_coco_dataset_labels(opt)
  number_of_class_labels = len(class_labels)

  output_size = number_of_class_labels + 5

  #  Attention: Some models are reversed!
  reverseModel = False

  strides = [8, 16, 32]

  if not all(opt.img_size % i == 0 for i in strides):
    assert False, 'Error: Please provide valid image size'

  if reverseModel:
    strides.reverse()
  feature_map_dimensions = [opt.img_size // stride for stride in strides]

  anchors = ([10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [
      116, 90, 156, 198, 373, 326])  # Take these from the <model>.yml in yolov5
  if reverseModel:
    anchors = anchors[::-1]

  anchor_grid = torch.tensor(anchors).float().view(3, -1, 1, 1, 2)

  Path(opt.model_output_directory).mkdir(parents=True, exist_ok=True)

  sample_input = torch.zeros((1, 3, opt.img_size, opt.img_size))
  checkInputs = [(torch.rand(1, 3, opt.img_size, opt.img_size),),
                 (torch.rand(1, 3, opt.img_size, opt.img_size),)]

  model = torch.load(opt.model_input_path, map_location=torch.device('cpu'))[
      'model'].float()

  model.eval()
  model.model[-1].export = True
  # Dry run, necessary for correct tracing!
  model(sample_input)

  ts = export_torchscript(model, sample_input, checkInputs,
                          f'{opt.model_output_directory}/{opt.model_output_name}.torchscript.pt')

  # Convert pytorch to raw coreml model
  modelSpec = convert_to_coreml_spec(ts, sample_input)
  add_output_meta_data(modelSpec, feature_map_dimensions, output_size)

  # Add export logic to coreml model
  builder = ct.models.neural_network.NeuralNetworkBuilder(spec=modelSpec)
  add_export_layer_to_coreml(
      opt, builder, anchor_grid, feature_map_dimensions, strides, number_of_class_labels)

  # Create nms logic
  nms_spec = create_nms_model_spec(
      builder.spec, number_of_class_labels, class_labels)

  # Combine model with export logic and nms logic
  combine_models_and_export(opt, builder.spec, nms_spec,
                            f'{opt.model_output_directory}/{opt.model_output_name}.mlmodel', opt.quantize)


if __name__ == '__main__':
  main()
