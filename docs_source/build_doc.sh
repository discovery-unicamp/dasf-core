#!/bin/bash

# source docs-venv/bin/activate
rm -rf ../docs
sphinx-build source/ ../docs
echo " " > ../docs/.nojekyll
# deactivate
