#!/bin/bash
#
version=$(python -c 'import platform; major, minor, patch = platform.python_version_tuple(); print(f"{major}.{minor}")')
filepath=$(pwd)
# parentname="$(dirname "$filepath")"
# echo "$parentname"
echo "$filepath" >>"$CONDA_PREFIX/lib/python$version/site-packages/tmp.pth"
