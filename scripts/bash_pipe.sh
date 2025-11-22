python exp_pipe.py \
--gpu 1 \
--model SAOT_Structured_Mesh_2D \
--n-hidden 192 \
--n-heads 8 \
--n-layers 8 \
--mlp_ratio 1 \
--lr 0.001 \
--max_grad_norm 0.1 \
--batch-size 1 \
--slice_num 64 \
--unified_pos 0 \
--ref 8 \
--eval 0 \
--save_name pipe_SAOT

