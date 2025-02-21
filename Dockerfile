# SPDX-FileCopyrightText: 2022 The eminus developers
# SPDX-License-Identifier: Apache-2.0
# Build everything using multi-stage builds
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS build

# Create a working directory and Python environment
WORKDIR /usr/app/
RUN uv venv /usr/app/venv/
ARG PATH="/usr/app/venv/bin:$PATH"

# Install Git to clone repositories and clean up afterwards
RUN apt-get update -y \
&& apt-get install -y git --no-install-recommends \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*

# Install Torch manually since we only want to compute on the CPU
RUN uv pip install torch --index-url https://download.pytorch.org/whl/cpu --no-cache-dir

# Install eminus with all extras available
# Use an editable installation so users can make changes on the fly
# We can pass the branch name when building the image but default to the main branch
ARG BRANCH=main
RUN git clone -b ${BRANCH} https://gitlab.com/wangenau/eminus.git \
&& uv pip install -e eminus/[all,dev] --no-cache-dir

# Set up the application stage
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim
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
CMD ["sh", "-c", "jupyter lab . --no-browser --ip 0.0.0.0 --port 8888"]
