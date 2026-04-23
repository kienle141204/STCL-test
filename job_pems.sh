#----------KPrompt-----------#
python main.py --conf conf/PEMS/kprompt.json --gpuid 0 --seed 23 --logname "kprompt_96" --backbone "stgnn"
python main.py --conf conf/PEMS/kprompt.json --gpuid 0 --seed 24 --logname "kprompt_96" --backbone "stgnn"
python main.py --conf conf/PEMS/kprompt.json --gpuid 0 --seed 25 --logname "kprompt_96" --backbone "stgnn"

python main.py --conf conf/PEMS/kprompt.json --gpuid 0 --seed 26 --logname "kprompt_96" --backbone "dcrnn"
python main.py --conf conf/PEMS/kprompt.json --gpuid 0 --seed 27 --logname "kprompt_96" --backbone "dcrnn"
python main.py --conf conf/PEMS/kprompt.json --gpuid 0 --seed 28 --logname "kprompt_96" --backbone "dcrnn"

python main.py --conf conf/PEMS/kprompt.json --gpuid 0 --seed 29 --logname "kprompt_96" --backbone "astgnn"
python main.py --conf conf/PEMS/kprompt.json --gpuid 0 --seed 30 --logname "kprompt_96" --backbone "astgnn"
python main.py --conf conf/PEMS/kprompt.json --gpuid 0 --seed 31 --logname "kprompt_96" --backbone "astgnn"

python main.py --conf conf/PEMS/kprompt.json --gpuid 0 --seed 32 --logname "kprompt_96" --backbone "tgcn"
python main.py --conf conf/PEMS/kprompt.json --gpuid 0 --seed 33 --logname "kprompt_96" --backbone "tgcn"
python main.py --conf conf/PEMS/kprompt.json --gpuid 0 --seed 34 --logname "kprompt_96" --backbone "tgcn"