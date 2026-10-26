#!/bin/bash
# Task 1.4 toy part. Run OUTSIDE your main repo:  bash dvc_toy.sh
set -x
mkdir dvc_toy && cd dvc_toy && git init && dvc init
printf "a,b\n1,2\n3,4\n" > data.csv
dvc add data.csv && git add data.csv.dvc .gitignore .dvc && git commit -m "v1 data"
printf "a,b\n1,2\n3,4\n5,6\n" > data.csv
dvc add data.csv && git commit -am "v2 data"
git checkout HEAD~1 data.csv.dvc && dvc checkout && cat data.csv     # back to v1
git checkout main data.csv.dvc 2>/dev/null || git checkout master data.csv.dvc; dvc checkout
dvc remote add -d localstore ../dvc_remote && dvc push
rm -rf data.csv .dvc/cache && dvc pull && cat data.csv
cat data.csv.dvc
