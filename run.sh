python train.py -data_pkl multi30k_en_de.pkl \
-output_dir output/lr_mul_0.5-scale_emb \
-embs_share_weight \
-proj_share_weight \
-label_smoothing \
-scale_emb_or_prj emb \
-lr_mul 0.25 \
-b 256 \
-gas 8 \
-warmup 4000 \
-epoch 600 \
-seed 0 \
-fr_warmup_epoch 601 


for SEED in 1 2 3
do
  python train.py -data_pkl m30k_deen_shr.pkl \
  -log m30k_deen_shr \
  -embs_share_weight \
  -proj_share_weight \
  -label_smoothing \
  -save_model best \
  -b 256 \
  -warmup 128000 \
  -epoch 400 \
  -seed ${SEED} \
  -fr_warmup_epoch 401 > vanilla_transformer_seed${SEED}_log 2>&1
done 


# for SEED in 1 2 3
# do
#   python train.py -data_pkl m30k_deen_shr.pkl \
#   -log m30k_deen_shr \
#   -embs_share_weight \
#   -proj_share_weight \
#   -label_smoothing \
#   -save_model best \
#   -b 256 \
#   -warmup 4000 \
#   -epoch 200 \
#   -seed ${SEED} \
#   -fr_warmup_epoch 201 > vanilla_transformer_seed${SEED}_log 2>&1
# done