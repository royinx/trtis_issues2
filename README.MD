# TritonIS issue <21.02>
nvcr.io/nvidia/tensorrt:21.02-py3 to compile
```
docker run  --rm \
            -it \
            --runtime nvidia \
            -v ${PWD}:/sharefolder \
            -w /sharefolder \
            nvcr.io/nvidia/tensorrt:21.02-py3 \
trtexec --onnx=model.onnx --saveEngine=model.plan --explicitBatch --fp16 --verbose --dumpOutput --minShapes='input':1x3x480x640 --optShapes='input':5x3x480x640  --maxShapes='input':64x3x480x640
```

```
docker-compose build && docker-compose up
```

Client controller:
```
docker exec -it controller bash

# batch size = 1 , Work
python3 test.py 1

# batch size = 5 , Raise error
python3 test.py 5 
```