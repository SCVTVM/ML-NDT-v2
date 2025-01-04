import tensorflow as tf
import subprocess
import os

def get_installed_cuda_version():
    try:
        cuda_version = subprocess.check_output(['nvcc', '--version']).decode()
        # Extract CUDA version from the output
        version_line = [line for line in cuda_version.split('\n') if 'release' in line]
        if version_line:
            return version_line[0].split('release ')[-1].split(',')[0]
    except Exception as e:
        print(f"Could not retrieve CUDA version: {e}")
    return "Not Found"

def get_installed_cudnn_version():
    cudnn_path = os.path.join(os.path.dirname(tf.sysconfig.get_lib()), 'include', 'cudnn_version.h')
    try:
        with open(cudnn_path, 'r') as f:
            for line in f:
                if '#define CUDNN_MAJOR' in line:
                    major_version = line.split()[-1]
                elif '#define CUDNN_MINOR' in line:
                    minor_version = line.split()[-1]
                elif '#define CUDNN_PATCHLEVEL' in line:
                    patch_version = line.split()[-1]
            return f"{major_version}.{minor_version}.{patch_version}"
    except Exception as e:
        print(f"Could not retrieve cuDNN version: {e}")
    return "Not Found"

def get_tensorflow_cuda_cudnn_versions():
    tf_cuda_version = tf.sysconfig.get_build_info()["cuda_version"]
    tf_cudnn_version = tf.sysconfig.get_build_info()["cudnn_version"]
    return tf_cuda_version, tf_cudnn_version

# Print CUDA, cuDNN versions
print("Installed CUDA version:", get_installed_cuda_version())
print("Installed cuDNN version:", get_installed_cudnn_version())

# Print TensorFlow expected CUDA and cuDNN versions
tf_cuda_version, tf_cudnn_version = get_tensorflow_cuda_cudnn_versions()
print("TensorFlow expects CUDA version:", tf_cuda_version)
print("TensorFlow expects cuDNN version:", tf_cudnn_version)