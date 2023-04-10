# DO NOT MODIFY THIS FILE
# This is the entry point for your submission.
# Changing this file will probably fail your submissions.

import os

import demo_BuildVillageHouse
import demo_CreateVillageAnimalPen
import demo_FindCave
import demo_MakeWaterfall

import train

# By default, only do testing
EVALUATION_STAGE = os.getenv("EVALUATION_STAGE", "testing")

# Training Phase
if EVALUATION_STAGE in ["all", "training"]:
    train.main()

# Testing Phase
if EVALUATION_STAGE in ["all", "testing"]:
    # demo_BuildVillageHouse.main()
    demo_CreateVillageAnimalPen.main()
    demo_FindCave.main()
    # demo_MakeWaterfall.main()
