To get ressources
```
salloc -c 8 -p gpu --gres=gpu:rtx8000 --time=14:00:00
salloc -c 8 -p gpu --gres=gpu:a100 --time=14:00:00
```

To activate the virtual env: `source .venv/bin/activate`

On a H100, the first benchmarks are working :
```
$ python benchmarking_script.py 
$ python benchmarking_script.py --d_model=1600 --d_ff=6400 --num_layers=48 --num_heads=25
```
this one gives a oom:
```
$ python benchmarking_script.py --d_model=2560 --d_ff=10240 --num_layers=32 --num_heads=32
```
Adding a `os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"` in the file did not help.

On a RTX8000
```
$ python benchmarking_script.py 
$ python benchmarking_script.py --d_model=1024 --d_ff=4096 --num_layers=24 --num_heads=16
```
! Std becomes huge!
Average time per step (forward + backward): 2.261189 seconds                                                                                                
Standard deviation: 0.548800 seconds
Not able to reproduce???

```
$ python benchmarking_script.py --d_model=1024 --d_ff=4096 --num_layers=24 --num_heads=16 --mixed_precision
```

To copy the memory snapshots:
```
scp lelarge@cleps.inria.fr:'/home/lelarge/GitHub/assignment2-systems/cs336_systems/benchmarking/*.pickle' .
```
Then go to the website: `https://docs.pytorch.org/memory_viz`
Useful blog: `https://pytorch.org/blog/understanding-gpu-memory-1/`

on H100
```
$ module load nsight-systems/2025.6.1
$ uv run nsys profile -o result python benchmarking_script.py
$ uv run nsys profile -o result2 python benchmarking_script.py --d_model=1600 --d_ff=6400 --num_layers=48 --num_heads=25 --mixed_precision --memory_profile
```

```
Generated at: 
/home/lelarge/GitHub/assignment2-systems/cs336_systems/benchmarking/result.nsys-rep
/home/lelarge/GitHub/assignment2-systems/cs336_systems/benchmarking/result2.nsys-rep
```

01/12
In order to be able to use triton, I created a conda env:
```
# before:
uv pip freeze > requirements.txt
# then
conda create -n cs336-syst python=3.11
conda activate cs336-syst
pip install -r requirements.txt
```


**Fused Softmax:**
- The .warmup() call already compiles the kernel with the specified BLOCK_SIZE and num_stages values:
```python
kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, 
                               BLOCK_SIZE=BLOCK_SIZE, num_stages=num_stages, 
                               num_warps=num_warps, grid=(1, ))
```
The returned kernel object is a pre-compiled version that no longer needs (or accepts) these metaparameters. You only pass the regular runtime arguments when launching it.
- to add numba
```
pip install numba
conda install -c conda-forge cudatoolkit-dev
```