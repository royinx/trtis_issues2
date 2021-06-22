import numpy as np
import sys

import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

def test_infer(model_name,
               dynamic,
               input_cfg,
               output_cfg,
               input_datas,
               headers=None,
               request_compression_algorithm=None,
               response_compression_algorithm=None):
    inputs = []
    outputs = [] 
    assert len(input_cfg) == len(input_datas)

    for idx, (input_layer, input_data) in enumerate(zip(input_cfg, input_datas)):
        if dynamic:
            input_layer["dims"].insert(0,len(input_data))

        inputs.append(httpclient.InferInput(input_layer["name"], input_layer["dims"], input_layer["data_type"].split("_")[-1]))
        inputs[idx].set_data_from_numpy(input_data, binary_data=False)

    for idx, output_layer in enumerate(output_cfg):
        outputs.append(httpclient.InferRequestedOutput(output_layer["name"], binary_data=False))

    query_params = {'test_1': 1, 'test_2': 2}

    results = triton_client.infer(
        model_name,
        inputs,
        outputs=outputs,
        query_params=query_params,
        headers=headers
        )

    rs = []
    for output_layer in results.get_response()["outputs"]:
        out_ = np.array(output_layer["data"]).reshape(output_layer["shape"])
        out_ = np.squeeze(out_,axis=0)
        rs.append(out_)

    return rs


if __name__ == '__main__':
    try:
        url = "triton:8000"
        triton_client = httpclient.InferenceServerClient(
            url=url, verbose=False)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)


    # Create the data for the two input tensors. Initialize the first
    # to unique integers and the second to all ones.
    

    batch_size = int(sys.argv[1])
    im = np.ones((batch_size,3,480,640),dtype=np.float32)
    im = np.ascontiguousarray(im)

    input_datas = [im]

    model_name = "model_x"
    assert any(model_['name'] == model_name for model_ in triton_client.get_model_repository_index()), \
        f'model "{model_name}" is not loaded in Triton server, check the server models and client config file'

    model_config = triton_client.get_model_config(model_name=model_name,
                                                  model_version=1)

    headers_dict = None
    dynamic = "dynamic_batching" in model_config.keys()
    results = test_infer(model_name,
                         dynamic,
                         model_config["input"],
                         model_config["output"],
                         input_datas, 
                         headers_dict)

    statistics = triton_client.get_inference_statistics(model_name=model_name,
                                                        headers=headers_dict)

    last_inf_time = statistics["model_stats"][0]["last_inference"]/1000
    from datetime import datetime
    print(datetime.utcfromtimestamp(last_inf_time).strftime('%Y-%m-%d %H:%M:%S'))