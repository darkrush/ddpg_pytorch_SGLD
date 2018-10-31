ENV_LIST='Hopper-v2 Walker2d-v2 Swimmer-v2 Reacher-v2 Humanoid-v2 Ant-v2 HumanoidStandup-v2 HalfCheetah-v2'
rm exp_list.txt
touch exp_list.txt
for env_name in $ENV_LIST
do 
    echo "python main.py  --env $env_name  --exp-name 1  --SGLD-mode 2  --num-pseudo-batches 10000000 --pool-mode 0  --pool-size 0                " >> exp_list.txt
    echo "python main.py  --env $env_name  --exp-name 2  --SGLD-mode 2  --num-pseudo-batches 1000000  --pool-mode 0  --pool-size 0                " >> exp_list.txt
    echo "python main.py  --env $env_name  --exp-name 3  --SGLD-mode 2  --num-pseudo-batches 100000   --pool-mode 0  --pool-size 0                " >> exp_list.txt
    echo "python main.py  --env $env_name  --exp-name 4  --SGLD-mode 2  --num-pseudo-batches 10000    --pool-mode 0  --pool-size 0                " >> exp_list.txt
    #echo "python main.py  --env $env_name  --exp-name 4  --SGLD-mode 0  --num-pseudo-batches 0        --pool-mode 0  --pool-size 0  --action-noise" >> exp_list.txt
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