import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException
import sys
import numpy as np
from process import preprocess


def triton_preprocess(url,
                      model):
    model_info = False
    verbose = False
    client_timeout = None
    ssl = False
    root_certificates = None
    private_key = None
    certificate_chain = None

    # Create server context
    try:
        triton_client = grpcclient.InferenceServerClient(
            url=url,
            verbose=verbose,
            ssl=ssl,
            root_certificates=root_certificates,
            private_key=private_key,
            certificate_chain=certificate_chain)
    except Exception as e:
        print("context creation failed: " + str(e))
        sys.exit()

    # Health check
    if not triton_client.is_server_live():
        print("FAILED : is_server_live")
        sys.exit(1)

    if not triton_client.is_server_ready():
        print("FAILED : is_server_ready")
        sys.exit(1)

    if not triton_client.is_model_ready(model):
        print("FAILED : is_model_ready")
        sys.exit(1)

    try:
        metadata = triton_client.get_model_metadata(model)
        print(metadata)
    except InferenceServerException as ex:
        if "Request for unknown model" not in ex.message():
            print("FAILED : get_model_metadata")
            print("Got: {}".format(ex.message()))
            sys.exit(1)
        else:
            print("FAILED : get_model_metadata")
            sys.exit(1)

    # Model configuration
    try:
        config = triton_client.get_model_config(model)
        if not (config.config.name == model):
            print("FAILED: get_model_config")
            sys.exit(1)
        print(config)
    except InferenceServerException as ex:
        print("FAILED : get_model_config")
        print("Got: {}".format(ex.message()))
        sys.exit(1)
    
    return triton_client, model_info, client_timeout

def triton_infer(model = None,
          client_timeout = False,
          model_info = False,
          num_stage = 4,
          input_img = None, 
          input_shape = None,
          mean = None,
          std = None,
          triton_client = None,
          mt_cls_output_blob_name = None,
          mt_reg_output_blob_name = None):
    if not input_img:
        print("FAILED: no input image")
        sys.exit(1)

    inputs = []
    outputs = []
    inputs.append(grpcclient.InferInput('data', [1, 3, 384, 672], "FP32"))
    for i in range(len(mt_cls_output_blob_name)):   
        outputs.append(grpcclient.InferRequestedOutput(mt_cls_output_blob_name[i]))
        outputs.append(grpcclient.InferRequestedOutput(mt_reg_output_blob_name[i]))
    
    input_image_buffer = preprocess(input_img,
                                    input_shape = input_shape,
                                    mean = mean,
                                    std = std)

    input_image_buffer = np.expand_dims(input_image_buffer, axis=0)
    inputs[0].set_data_from_numpy(input_image_buffer)

    print("Invoking inference...")
    results = triton_client.infer(model_name=model,
                                  inputs=inputs,
                                  outputs=outputs,
                                  client_timeout=client_timeout)
    
    if model_info:
        statistics = triton_client.get_inference_statistics(model_name=model)
        if len(statistics.model_stats) != 1:
            print("FAILED: get_inference_statistics")
            sys.exit(1)
        # print(statistics)
    print("load model done")

    cls_score_list = []
    bbox_preds_list = []
    for i in range(num_stage):
        cls_score_list.append(results.as_numpy(mt_cls_output_blob_name[i]))
        bbox_preds_list.append(results.as_numpy(mt_reg_output_blob_name[i]))
    
    return cls_score_list, bbox_preds_list


