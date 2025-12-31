# Copyright (c) ModelZoo. 2025-2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from ultralytics.data.split_dota import split_test, split_trainval


def split_dota(origin_path, saved_path):
    # Split train and val set, with labels.
    split_trainval(
        data_root=origin_path,
        save_dir=saved_path,
        rates=[0.5, 1.0, 1.5],  # multiscale
        gap=500,
    )
    # Split test set, without labels.
    split_test(
        data_root=origin_path,
        save_dir=saved_path,
        rates=[0.5, 1.0, 1.5],  # multiscale
        gap=500,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split DOTA dataset")
    parser.add_argument(
        "--origin_path",
        type=str,
        default="datasets/DOTAv1/",
        help="Path to original dataset",
    )
    parser.add_argument(
        "--saved_path",
        type=str,
        default="datasets/DOTAv1-split/",
        help="Path to save split dataset",
    )
    args = parser.parse_args()

    split_dota(args.origin_path, args.saved_path)