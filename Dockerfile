FROM python:3.10-slim

# Install necessary build dependencies and cleanup afterwards
# git is needed to clone repos
# libglfw3 is needed for pyflosic2
# gfortran and python3-dev are needed to compile fodmc (installed with pyflosic2)
RUN apt-get update -y \
    && apt-get install gfortran git libglfw3 python3-dev -y --no-install-recommends \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install jupyter with a pinned ipywidgets version
# (see https://github.com/nglviewer/nglview/pull/1033)
# Somehow numpy has to be installed before installing pyflosic2
RUN pip install notebook ipywidgets==7.* numpy --no-cache-dir

# Fix to install pyflosic2 with a new pyscf version
RUN git clone https://gitlab.com/opensic/pyflosic2.git \
    && sed -i "s/pyscf==1.7.6.post1/pyscf/g" pyflosic2/setup.py \
    && pip install pyflosic2/ --no-cache-dir \
    && rm -rf pyflosic2

# Install eminus with all extras available
RUN git clone https://gitlab.com/wangenau/eminus.git \
    && pip install -e eminus/[all] --no-cache-dir

# Run jupyter
CMD ["sh", "-c", "jupyter notebook /eminus --no-browser --allow-root --ip 0.0.0.0 --port 8888"]
