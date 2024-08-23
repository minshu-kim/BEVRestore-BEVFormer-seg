# Prerequisites

**Please ensure you have prepared the environment and the nuScenes dataset.**

# Train and Test

Train BEVFormer using 0,1,2,3 GPUs
```
./scripts/train-bevformer.sh 0,1,2,3
```

Train BEVFormer w/ BEVRestore using 0,1,2,3 GPUs
```
./scripts/train-with-bevrestore.sh 0,1,2,3
```


Eval BEVFormer with a GPU
```
./scripts/eval.sh
```
