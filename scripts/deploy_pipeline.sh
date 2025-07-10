# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
set -e

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

read -s -p "Enter your Hugging Face API key: " HF_KEY
echo
export HUGGING_FACE_HUB_TOKEN="$HF_KEY"

conda_env="cosmos-predict1"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is required but was not found. Please install conda and rerun." >&2
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -q "^${conda_env}$"; then
    echo "Creating conda environment ${conda_env}..."
    conda env create --file cosmos-predict1.yaml
fi

conda activate "$conda_env"

echo "\nInstalling dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pip install gradio
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/python3.10
pip install transformer-engine[pytorch]==1.12.0
git clone https://github.com/NVIDIA/apex || true
CUDA_HOME=$CONDA_PREFIX pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
  --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./apex

# Login to Hugging Face after installing huggingface-hub
huggingface-cli login --token "$HF_KEY"

echo "\nAvailable Cosmos-Predict1 models:\n"
grep "\* \[Cosmos" README.md | sed 's/^* //'

python3 scripts/test_environment.py || true

echo "\nDeployment complete."
