# Making Instructions

To `make` the `pc_distance` functions, here is the process I followed:

1. Make sure you have the `requirements.txt` installed into system Python
2. Get the TensorFlow framework directory

```bash
python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())'
# This output: /opt/conda/lib/python3.7/site-packages/tensorflow_core
```

3. Update the `makefile` to match the TensorFlow framework directory:

```
tf_lib = /usr/local/lib/python3.5/dist-packages/tensorflow
tf_inc = /usr/local/lib/python3.5/dist-packages/tensorflow/include
```

4. Please also update the other `makefile` variables as well
    - You can use `whereis nvcc` to verify the `nvcc` location

```
cuda_inc = /usr/local/cuda-11.0/include/
cuda_lib = /usr/local/cuda-11.0/lib64/
nvcc = /usr/local/cuda-11.0/bin/nvcc
```

5. Now, `cd` to `pc_distance` and run `make`... best of luck!

If you error out (like I did) with: `/usr/bin/ld: cannot find -ltensorflow_framework`

Then per [source](https://github.com/bgshih/aster/issues/56) let's:
1. `cd` to the TensorFlow framework directory
    - Follow instructions above to find this directory
2. Run `ln -s libtensorflow_framework.so.1 libtensorflow_framework.so`
3. Retry `make`ing again from the `pc_distance` directory
