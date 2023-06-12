# Human-Guided-Phasic-Policy-Gradient-in-Minecraft

## Installation

To run this project, you need to install the following dependencies:

- Pre-requirements for [MineRL](https://minerl.readthedocs.io/en/v1.0.0/tutorials/index.html)
- Anaconda/Miniconda

Once you have the dependencies installed, create the environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
```

## Downloading Model Files and Weights
Before running the code, you need to download the foundation model files and weights from [OpenAI VPT repository](https://github.com/openai/Video-Pre-Training).

## Downloading the BASALT dataset

You can find the index files containing all the download URLs for the full BASALT dataset in the [OpenAI VPT repository at the bottom](https://github.com/openai/Video-Pre-Training#basalt-2022-dataset).

We have included a download utility (`utils/download_dataset.py`) to help you download the dataset. You can use it with the index files from the OpenAI VPT repository. For example, if you download the FindCave dataset index file, named `find-cave-Jul-28.json`, you can download first 100 demonstrations to `MineRLBasaltFindCave-v0` directory with:

```
python download_dataset.py --json-file find-cave-Jul-28.json --output-dir MineRLBasaltFindCave-v0 --num-demos 100
```

Basic dataset statistics (note: one trajectory/demonstration may be split up into multiple videos):

```
Size  #Videos  Name
--------------------------------------------------
146G  1399     MineRLBasaltBuildVillageHouse-v0 C:\Users\Dvalv\Documents\Github\Playing-video-games\data\MineRLBasaltBuildVillageHouse-v0\build-house-Jul-28.json
165G  2833     MineRLBasaltCreateVillageAnimalPen-v0 C:\Users\Dvalv\Documents\Github\Playing-video-games\data\MineRLBasaltCreateVillageAnimalPen-v0\pen-animals-Jul-28.json
165G  5466     MineRLBasaltFindCave-v0
C:\Users\Dvalv\Documents\Github\Playing-video-games\data\MineRLBasaltFindCave-v0\find-cave-Jul-28.json
175G  4230     MineRLBasaltMakeWaterfall-v0
C:\Users\Dvalv\Documents\Github\Playing-video-games\data\MineRLBasaltMakeWaterfall-v0\waterfall-Jul-28.json
```

