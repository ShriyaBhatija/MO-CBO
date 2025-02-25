FROM --platform=linux/amd64 ubuntu:18.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# Install wget to fetch Miniconda
RUN apt-get update && \
    apt-get install -y wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install basic utilities and build tools
RUN apt-get update && \
    apt-get install -y wget build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda (x86_64 installer)
RUN MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-py37_22.11.1-1-Linux-x86_64.sh" && \
    wget $MINICONDA_URL -O miniconda.sh && \
    mkdir -p /root/.conda && \
    bash miniconda.sh -b -p /root/miniconda3 && \
    rm -f miniconda.sh

RUN conda --version

# Set the working directory inside the container
WORKDIR /app

# Copy your environment file into the container
COPY environment.yml .

# Create the Conda environment
RUN conda env create -f environment.yml --verbose --name mo_cbo

# Set conda configuration to use the linux-64 subdir (for x86_64 packages)
#RUN conda config --env --set subdir linux-64

# Optionally initialize conda for bash (if you need it for interactive shells)
RUN conda init bash

# Set default shell to bash
SHELL ["/bin/bash", "-c"]

# Copy all your project files into the container
COPY . .

# Set the entrypoint to activate the Conda environment and run your script

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "mo_cbo"]
CMD ["python", "main.py"]
