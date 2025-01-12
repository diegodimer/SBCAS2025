#!/bin/bash

pdflatex main
bibtex main
pdflatex main
rm -rf main.aux main.bbl main.blg main.log main.fdb_latexmk main.fls