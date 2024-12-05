#Demo train
#python train.py --device=mps --dtype=float32 --compile=False --vocab_source=llama2 --vocab_size=32000 --max_iters=100 --eval_iters=10 --batch_size=8

python train.py \
  --vocab_source=llama2 \
  --vocab_size=32000 \
  --batch_size=128 \
  --gradient_accumulation_steps=1 \
  --max_iters=100000 \
  --eval_interval=1000 \
  --device=mps \
  --dtype=float32 \
  --compile=False
