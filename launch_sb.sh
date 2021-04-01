#!bin/bash


<<<<<<< HEAD
# ==================== OBSERVATIONS ====================

# IF INITIAL BALANCE IS TOO LOW: AGENT BUYS STOCK AND WAITS
# EVEN WITH HIGHER, RANDOMIZED BALANCE, AGENTS BUYS STOCK AND WAITS
# TESTING WITH SAME PARAMETERS BUT BITCOIN 
# GETTING SIMILAR RESULTS: NO ACTIONS 
# CHANGING REWARD TO DISTANCE BETWEEN NET WORTH AND 

# ======================================================

# python run_experiment.py --env=BC_Ac --layers="512,512" --train_steps=25
python run_experiment.py --env=BC_I --layers="512,512" --train_steps=20
# python run_experiment.py --env=BC_I --layers="512,512" --train_steps=20 
# python run_eval.py --env=BC_I --agent_id=7 --deterministic
# python run_eval.py --env=BC_I --agent_id=7 --do_plot
# shutdown 
# python run_experiment.py --env=BC_I --layers="256,256" --train_steps=10
# python run_experiment.py --env=BC_CARG --layers="256,256" --train_steps=50
# python run_experiment.py --env=BC_CARG --layers="1024,512" --train_steps=50
# python run_eval.py --random_agent --env_id=BC_I
# python run_eval.py --deterministic --agent_id=0 --env_id=BC_I

# python run_experiment.py --env=MCD
# python run_eval.py --random_agent --env_id=MCD
# python run_eval.py --deterministic --agent_id=2 --env_id=MCD
# python run_eval.py --agent_id=2 --env_id=MCD --do_plot

# python run_experiment.py --env=MCD --layers="512,512"
# python run_eval.py --deterministic --agent_id=3 --env_id=MCD
# python run_eval.py --agent_id=4 --env_id=MCD
# python run_eval.py --agent_id=4 --env_id=MCD --do_plot
=======
python sb_agent.py --d
python sb_agent.py
python sb_agent.py --rand 
>>>>>>> parent of b420023... commit before refactoring archi
