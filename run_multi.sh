ENV_LIST='Ant-v2 HalfCheetah-v2 Hopper-v2 Humanoid-v2 HumanoidStandup-v2 InvertedDoublePendulum-v2 InvertedPendulum-v2 Reacher-v2 Swimmer-v2 Walker2d-v2'
COEF_LIST='1 0.1 0.01 0.001 0.0001'

for env_name in $ENV_LIST
do 
    for coef in $COEF_LIST
    do
        python main.py --env $env_name --SGLD-coef $coef --noise-decay 0 --pool-size 10 --lr-decay 0 --pool-mode 3 -rand_seed 666
        
    done
        python main.py --env $env_name --SGLD-coef 0.0   --noise-decay 1 --pool-size 0  --lr-decay 0 --pool-mode 0  --action-noise -rand_seed 666
done