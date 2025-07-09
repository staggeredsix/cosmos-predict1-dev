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

echo "\nInstalling dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install --upgrade torch torchvision

# Login to Hugging Face after installing huggingface-hub
huggingface-cli login --token "$HF_KEY" --non-interactive

echo "\nAvailable Cosmos-Predict1 models:\n"
grep "\* \[Cosmos" README.md | sed 's/^* //'

python3 scripts/test_environment.py || true

echo "\nDeployment complete."
