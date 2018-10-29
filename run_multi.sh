ENV_LIST='Walker2d-v2 Ant-v2 Hopper-v2 Swimmer-v2 HumanoidStandup-v2 HalfCheetah-v2 Humanoid-v2 Reacher-v2 InvertedDoublePendulum-v2 InvertedPendulum-v2'
rm exp_list.txt
touch exp_list.txt
for env_name in $ENV_LIST
do 
    echo "python main.py  --env $env_name  --exp-name 1  --SGLD-mode 2  --SGLD-coef 0.00    --pool-mode 0  --pool-size 0" >> exp_list.txt
    echo "python main.py  --env $env_name  --exp-name 2  --SGLD-mode 2  --SGLD-coef 0.1     --pool-mode 0  --pool-size 0" >> exp_list.txt
    echo "python main.py  --env $env_name  --exp-name 3  --SGLD-mode 2  --SGLD-coef 0.01    --pool-mode 0  --pool-size 0" >> exp_list.txt
    echo "python main.py  --env $env_name  --exp-name 4  --SGLD-mode 2  --SGLD-coef 0.001   --pool-mode 0  --pool-size 0" >> exp_list.txt
    echo "python main.py  --env $env_name  --exp-name 5  --SGLD-mode 2  --SGLD-coef 0.0001  --pool-mode 0  --pool-size 0" >> exp_list.txt
    echo "python main.py  --env $env_name  --exp-name 6  --SGLD-mode 0  --SGLD-coef 0       --pool-mode 0  --pool-size 0  --action-noise" >> exp_list.txt
done
export CUDA_VISIBLE_DEVICES=0
cat exp_list.txt | while read line 
do
    count=$[count+1]
    
    if [ "$CUDA_VISIBLE_DEVICES" = 0 ]; then
        export CUDA_VISIBLE_DEVICES=1
    else
        export CUDA_VISIBLE_DEVICES=0
    fi
    
    $line &
    
    if [ "$count" = 4 ]; then
        wait
        count=0
    fi
done