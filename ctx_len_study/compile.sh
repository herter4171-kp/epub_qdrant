#!/bin/bash
# Compile ctx_len_study
set -e
cd $(dirname "$0")
echo 'Compiling ctx_len_study.md...'
echo '# Context Window Length Study' > ctx_len_study.md
echo '' >> ctx_len_study.md
cat intro.md >> ctx_len_study.md
cat 2010__03768.md >> ctx_len_study.md
cat 2201__11903.md >> ctx_len_study.md
cat 2203__11171.md >> ctx_len_study.md
cat 2205__00445.md >> ctx_len_study.md
cat 2205__10625.md >> ctx_len_study.md
cat 2205__12255.md >> ctx_len_study.md
cat 2206__05802.md >> ctx_len_study.md
cat 2206__08853.md >> ctx_len_study.md
cat 2206__10498.md >> ctx_len_study.md
cat 2207__01206.md >> ctx_len_study.md
echo '' >> ctx_len_study.md
echo 'Done: ctx_len_study.md' 