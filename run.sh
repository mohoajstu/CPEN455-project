# Run the code
! python pcnn_train.py \
--batch_size 16 \
--sample_batch_size 16 \
--sampling_interval 25 \
--save_interval 25 \
--dataset cpen455 \
--nr_resnet 1 \
--lr_decay 0.999995 \
--max_epochs 1 \
--en_wandb True \
--tag "cond-pcnn-test-run"