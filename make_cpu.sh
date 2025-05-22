#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --account=bsc85
#SBATCH --qos=gp_bsccs
#SBATCH --job-name=pytorch_make
#SBATCH --error=err_cpu.txt
#SBATCH --output=out_cpu.txt
#SBATCH --exclusive
#SBATCH --hint=nomultithread


module purge
ml torchcpu


export CC=gcc
export DEBUG=1
export MAX_JOBS=112

export USE_CUDA=0
export USE_CUDNN=0
export USE_CUSPARSELT=0
export USE_CUDSS=0
export USE_CUFILE=0
export USE_ITT=0
export USE_SYSTEM_NCCL=0

export MOSE_ROOT=${HOME}/mpi_offload

export MKL_ROOT=$MKL_HOME
export MKL_INCLUDE=$MKL_ROOT/include
export MKL_LIBRARY=$MKL_ROOT/lib/intel64
export CMAKE_INCLUDE_PATH=$MKL_INCLUDE:$CMAKE_INCLUDE_PATH
export CMAKE_LIBRARY_PATH=$MKL_LIBRARY:$CMAKE_LIBRARY_PATH
export USE_STATIC_MKL=1
export BLAS=MKL
export MKL_THREADING=SEQ

export USE_DISTRIBUTED=1
export USE_MPI=1
export USE_TENSORPIPE=1
export USE_GLOO=0

export CMAKE_PREFIX_PATH=":${CMAKE_PREFIX_PATH}:${MKL_HOME}"
# export CMAKE_INSTALL_PREFIX="$( pwd )/install"
python setup.py install --user
# python setup.py develop --user
# python3 -m pip install --user -v --editable . install


