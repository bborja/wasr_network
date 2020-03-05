# Water-obstacle Separation and Refinement network (WaSR)

## Running in a Docker image with NVIDIA GPU support (requires Docker 19.03+)
```
docker build -t wasr .
docker run -it --rm --gpus all -v $(pwd)/:/wasr wasr:latest bash
python wasr_inference_noimu_general.py --img-path example_1.jpg
```

## No-IMU Version 
This architecture does not incorporate IMU information. The ARM and FFM modules are used to fuse encoder and decoder features.
Novel water-separation loss is applied early in the encoder (res4 block) to force-separate water pixels from obstacle pixels.

### wasr_train_noimu.py
Use this function to train the network.

(to-do: explain parameters)

### wasr_inference_noimu_general.py
This function can be used to perform inference on a single image. Example usage:
```
python wasr_inference_noimu_general.py --img-path example_1.jpg
```
The above command will take image <i>example_1.jpg</i> from folder <i>test_images/</i> and segment it. The segmentation result will be saved in the <i>output/</i> folder.
<table>
<tr>
 <td>Example input image</td> <td>Example segmentation output</td>
</tr>
<tr>
 <td><img src="test_images/example_1.jpg"></td> <td><img src="output/output_mask_1.png"></td>
</tr>
</table>

### wasr_inferences_imu.sh
Use this bash script to run the inference (<i>wasr_inference_noimu.py</i>) on MODD2 dataset.

## IMU Version (TO-DO)
....
