CUDA_VISIBLE_DEVICES=0 python -u train_Trans_teacher_21_mscmr.py    --base_lr   0.005       --num_classes 4      &
CUDA_VISIBLE_DEVICES=1 python -u train_Trans_teacher_21_mscmr.py    --base_lr   0.01        --num_classes 4         &
CUDA_VISIBLE_DEVICES=3 python -u train_Trans_teacher_21_mscmr.py    --base_lr   0.001       --num_classes 4        &
CUDA_VISIBLE_DEVICES=4 python -u train_Trans_teacher_21_mscmr.py    --base_lr   0.0005       --num_classes 4       &
CUDA_VISIBLE_DEVICES=5 python -u train_Trans_teacher_21_mscmr.py    --base_lr   0.0001       --num_classes 4       &
CUDA_VISIBLE_DEVICES=6 python -u train_Trans_teacher_21_mscmr.py   --seed 2023  --base_lr   0.005       --num_classes 4      









