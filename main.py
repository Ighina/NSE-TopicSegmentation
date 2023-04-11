# main.py
import torch
from lightning.pytorch.cli import ArgsType, LightningCLI

from models.lightning_model import TextSegmenter
from DataModule import TopSegDataModule


def cli_main(args: ArgsType = None):
    cli = LightningCLI(TextSegmenter, TopSegDataModule, args=args)

if __name__ == "__main__":
    cli_main()