import onnx_graphsurgeon as gs
import onnx
import numpy as np


def rebuild_defect_onnx_model(onnx_path, dst_onnx_path):
    graph = gs.import_onnx(onnx.load(onnx_path))

    maxOutObj = 1000
    decode_attrs = {'clsNum': 1,
                    'inputH': 544,
                    'inputW': 960,
                    'maxOutObj': maxOutObj,
                    'confThreshold': 0.6,
                    'strides': [8, 16, 32, 64, 128],
                    'base_lens': [16, 32, 64, 128, 256]}

    decode_output = gs.Variable(name='decode_output', dtype=np.float32, shape=(6001, 1, 1))
    node = gs.Node(op='DecodeFoveaBboxPlugin_CUSTOM',
                   name="DecodeFoveaBbox",
                   attrs=decode_attrs,
                   inputs=graph.outputs,
                   outputs=[decode_output])
    graph.nodes.append(node)

    nms_output = gs.Variable(name='selected_indices', dtype=np.int64, shape=(1000, 3))
    node = gs.Node(op='mmcvNonMaxSuppression',
                   name="NMS",
                   attrs={'center_point_box': 1,
                          'max_output_boxes_per_class': maxOutObj,
                          'iou_threshold': 0.8,
                          'score_threshold': 0.6,
                          'offset': 0},
                   inputs=[decode_output],
                   outputs=[nms_output])
    graph.nodes.append(node)

    graph = gs.Graph(graph.nodes, inputs=graph.inputs, outputs=[decode_output, nms_output])
    # graph = gs.Graph(graph.nodes, inputs=graph.inputs, outputs=[decode_output])
    graph.cleanup()
    onnx.save(gs.export_onnx(graph), dst_onnx_path)
    print(f'model has been saved to {dst_onnx_path}')


def rebuild_human_aux_onnx_model(onnx_path, dst_onnx_path):
    graph = gs.import_onnx(onnx.load(onnx_path))

    maxOutObj = 1000
    decode_attrs = {'clsNum': 80,
                    'inputH': 544,
                    'inputW': 960,
                    'maxOutObj': maxOutObj,
                    'confThreshold': 0.65,
                    'whRatioClip': 16 / 1000,
                    'scalesPerOctave': 3,
                    'octaveBaseScale': 4.,
                    'ratios': [0.5, 1.0, 2.0],
                    'strides': [8, 16, 32, 64, 128],
                    'centerH': 0.,
                    'centerW': 0.,
                    'centerOffsetH': 0.,
                    'centerOffsetW': 0.}

    inputs = graph.outputs

    decode_output = gs.Variable(name='decode_output', dtype=np.float32, shape=(6001, 1, 1))
    node = gs.Node(op='DecodeBboxWithAnchorPlugin_CUSTOM',
                   name="DecodeBboxWithAnchor",
                   attrs=decode_attrs,
                   inputs=inputs,
                   outputs=[decode_output])
    graph.nodes.append(node)

    nms_output = gs.Variable(name='selected_indices', dtype=np.int64, shape=(1000, 3))
    node = gs.Node(op='mmcvNonMaxSuppression',
                   name="NMS",
                   attrs={'center_point_box': 1,
                          'max_output_boxes_per_class': maxOutObj,
                          'iou_threshold': 0.75,
                          'score_threshold': 0.65,
                          'offset': 0},
                   inputs=[decode_output],
                   outputs=[nms_output])
    graph.nodes.append(node)

    graph = gs.Graph(graph.nodes, inputs=graph.inputs, outputs=[decode_output, nms_output])
    # graph = gs.Graph(graph.nodes, inputs=graph.inputs, outputs=[decode_output])
    graph.cleanup()
    onnx.save(gs.export_onnx(graph), dst_onnx_path)
    print(f'model has been saved to {dst_onnx_path}')


if __name__ == '__main__':
    origin_path="./onnx_model/yolox_l_20220525_finetune.onnx"
    dest_path="./onnx_model/yolox_l_20220525_finetune_add.onnx"
    rebuild_human_aux_onnx_model(origin_path,dest_path)
