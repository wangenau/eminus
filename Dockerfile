# Build everything using multi-stage builds
FROM python:3.10-slim as build

# Create a working directory and Python environment
WORKDIR /usr/app/
RUN python -m venv /usr/app/venv/
ARG PATH="/usr/app/venv/bin:$PATH"

# Install Git to clone repositories and clean up afterwards
RUN apt-get update -y \
&& apt-get install -y git --no-install-recommends \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*

# Install Jupyter with a pinned IPyWidgets version (https://github.com/nglviewer/nglview/pull/1033)
RUN pip install notebook ipywidgets==7.* --no-cache-dir

# Fix to install PyFLOSIC2 with a new PySCF version
# Install without dependencies since only the PyCOM method will be used
# Only ASE is required to run PyFLOSIC2
RUN git clone https://gitlab.com/opensic/pyflosic2.git \
&& pip install ase pyflosic2/ --no-cache-dir --no-deps \
&& rm -rf pyflosic2/

# Install eminus with all extras available (PyFLOSIC2 is already installed for the fods extra)
# Use an editable installation so users can make changes on the fly
RUN git clone https://gitlab.com/wangenau/eminus.git \
&& pip install -e eminus/[libxc,viewer,dev] --no-cache-dir

# Set up the application stage
FROM python:3.10-slim
LABEL maintainer="wangenau"

# Ensure that no root rights are being used, copy the environment and eminus source
RUN addgroup --system eminus \
&& adduser --system --group eminus \
&& mkdir /usr/app/ \
&& chown eminus:eminus /usr/app/
WORKDIR /usr/app/
COPY --chown=eminus:eminus --from=build /usr/app/venv/ ./venv/
COPY --chown=eminus:eminus --from=build /usr/app/eminus/ ./eminus/

# Set the working directory and set variables
WORKDIR /usr/app/eminus/
ENV PATH="/usr/app/venv/bin:$PATH"
ENV JUPYTER_PLATFORM_DIRS=1

# Set user, expose port, and run Jupyter
USER eminus
EXPOSE 8888
CMD ["sh", "-c", "jupyter notebook . --no-browser --ip 0.0.0.0 --port 8888"]
