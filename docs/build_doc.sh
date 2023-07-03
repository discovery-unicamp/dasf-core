#!/bin/bash

make clean
sphinx-build source/ .
echo " " > ../docs/.nojekyll