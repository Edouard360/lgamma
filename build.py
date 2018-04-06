import os
import torch
from torch.utils.ffi import create_extension
import pdb
import argparse

parser = argparse.ArgumentParser(description='GPU build script')
parser.add_argument('--cuda-path', type=str, default='/usr/local/cuda-7.5/targets/x86_64-linux/lib/', help='Path to CUDA binaries (try `locate libcudart.so`)')
args = parser.parse_args()

sources = ['src/functions.c', 'src/internals_s.c']
headers = ['src/functions.h', 'src/internals_s.h']
defines = []
with_cuda = False
extra_objects = []
libraries = []

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['src/functions_cuda.c']
    headers += ['src/functions_cuda.h']
    defines.append(('WITH_CUDA', None))
    # This was causing trouble on AWS server (p2.xlarge) with cuda
    # AMI deep learning Ubuntu
    #libraries += ["cudart", "cudadevrt"] 
    
    with_cuda = True
    extra_objects = ['src/functions.link.cu.o', 'src/internals.cu.o', 'src/functions_cuda_kernel.cu.o']

this_file = os.path.dirname(os.path.realpath(__file__))
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

ffi = create_extension(
    '_ext.functions',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    library_dirs=[args.cuda_path, os.path.join(this_file, 'src/')],
    libraries=libraries,
    extra_compile_args=["-std=gnu11"],
    extra_objects=extra_objects
)

if __name__ == '__main__':
    ffi.build()
