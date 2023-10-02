# Build everything using multi-stage builds
FROM python:3.11-slim as build

# Create a working directory and Python environment
WORKDIR /usr/app/
RUN python -m venv /usr/app/venv/
ARG PATH="/usr/app/venv/bin:$PATH"

# Install Git to clone repositories and clean up afterwards
RUN apt-get update -y \
&& apt-get install -y git --no-install-recommends \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*

# Install Torch manually since we only want to compute on the CPU
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu --no-cache-dir

# Install eminus with all extras available
# Use an editable installation so users can make changes on the fly
RUN git clone https://gitlab.com/wangenau/eminus.git \
&& pip install -e eminus/[all,dev] --no-cache-dir

# Set up the application stage
FROM python:3.11-slim
LABEL maintainer="wangenau"

# Ensure that no root rights are being used, copy the environment and eminus source
RUN addgroup --system eminus \
&& adduser --system --group eminus \
&& mkdir /usr/app/ \
&& chown eminus:eminus /usr/app/ \
&& mkdir /nonexistent/ \
&& chown eminus:eminus /nonexistent/
WORKDIR /usr/app/
COPY --chown=eminus:eminus --from=build /usr/app/ /usr/app/

# Set the working directory and set variables
WORKDIR /usr/app/eminus/
ENV PATH="/usr/app/venv/bin:$PATH"
ENV JUPYTER_PLATFORM_DIRS=1

# Set user, expose port, and run Jupyter
USER eminus
EXPOSE 8888
CMD ["sh", "-c", "jupyter notebook . --no-browser --ip 0.0.0.0 --port 8888"]
