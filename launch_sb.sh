#!bin/bash


# ==================== OBSERVATIONS ====================

# python sb_agent.py --train --name="net_worth_state10"
# python sb_agent.py --train --dqn --name="net_worth_state10" 

python run_experiment.py --env=BC_S --layers="512,512" --train_steps=20
python run_experiment.py --env=BC_S --layers="256,256" --train_steps=20
python run_experiment.py --env=BC_S --layers="1024,1024" --train_steps=20
python run_eval.py --env=BC_S --random_agent
python run_eval.py --env=BC_S --agent_id=0 --deterministic
python run_eval.py --env=BC_S --agent_id=1 --deterministic
python run_eval.py --env=BC_S --agent_id=2 --deterministic


# python run_experiment.py --env=MCD
# python run_eval.py --random_agent --env_id=MCD
# python run_eval.py --deterministic --agent_id=2 --env_id=MCD
# python run_eval.py --agent_id=2 --env_id=MCD --do_plot

# python run_experiment.py --env=MCD --layers="512,512"
# python run_eval.py --deterministic --agent_id=3 --env_id=MCD
# python run_eval.py --agent_id=4 --env_id=MCD
# python run_eval.py --agent_id=4 --env_id=MCD --do_plot

# python sb_agent.py --d --name="ReLU64M"
# python sb_agent.py --name="ReLU64M"
# python sb_agent.py --rand 

