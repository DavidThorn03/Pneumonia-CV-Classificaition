#!/bin/bash

SRC=~/projects/pneumonia
DEST="/mnt/c/Users/davyt/OneDrive - Technological University Dublin/College/Sem8/ComputerVision/Pneumonia-CV-Classificaition"

echo "Syncing project to Windows..."

rsync -av --exclude='chest_xray/' \
    --exclude='chest_xray.zip' \
  "$SRC/" "$DEST/"

echo "Done."