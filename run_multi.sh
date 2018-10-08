for i in `seq 1 10`
do 
python main.py --env Humanoid-v2      --SGLD-coef 0.01  --noise-decay 0  --pool-size 10  --lr-decay 0   --pool-mode 3 

#python main.py --env HalfCheetah-v2      --SGLD-coef 0.01  --noise-decay 0  --pool-size 10  --lr-decay 10  --pool-mode 3 
#python main.py --env HalfCheetah-v2      --SGLD-coef 0.0    --noise-decay 1  --pool-size 0   --lr-decay 10  --pool-mode 0  --action-noise
done 

for i in `seq 1 10`
do 
python main.py --env Humanoid-v2      --SGLD-coef 0.0    --noise-decay 1  --pool-size 0   --lr-decay 0   --pool-mode 0  --action-noise
done 
