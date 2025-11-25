To activate the virtual env: `source .venv/bin/activate`

On a H100, the first benchmarks are working :
```
$ python benchmarking_scirpt.py 
$ python benchmarking_scirpt.py --d_model=1600 --d_ff=6400 --num_layers=48 --num_heads=25
```
this one gives a oom:
```
$ python benchmarking_scirpt.py --d_model=2560 --d_ff=10240 --num_layers=32 --num_heads=32
```
Adding a `os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"` in the file did not help.