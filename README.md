# SBCAS2025_Diego_PreTrainingBias

This repository contains the code and data for the SBCAS 2025 project on pre-training bias.

## Introduction
This project aims to explore the impact of pre-training bias in machine learning models. Four experiment files are provided in the `code` folder, each generating a set of plots and latex tables to report accuracy, F1-Score and information on the pre-training bias metrics and the algorithms' performance.

To extend to a new Dataset, one simply need to create a new file under `Datasets/` providing all the necessary input variables. In this repository is also the article `.tex` file.

## Installation
To install the necessary dependencies, run:
```bash
pip install -r fairnessinsight/requirements.txt
```

To build the PDF from the article, you need `pdflatex` and `bibtex`. The script `gen_pdf.sh` can be used to generate the pdf.

## Usage
To run the experiments, run the files with the suffix `-Experiment` in the `code` folder.

