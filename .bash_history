apt update && apt install -y tmux
nohup env PYTHONPATH=. python inference/dit360_edit_no_mask.py > inference/outputs/edit_run.log 2>&1 &
disown
kill -9 50
nvidia-smi
python inference/dit360_edit_no_mask.py
ls
pwd
PYTHONPATH=. python inference/dit360_edit_no_mask.py
git pull
git stash .
git stash
git pull
git stash pop
git fetch origin
git reset --hard origin/main
git clean -fd
exit
