# learn_exact
厳密計算の実装\
可視変数の数は4次元固定

ex)
```
$ ./learn_exact --data_num 6000 --h_dim 5 --nabla 0.01 --step 1000 --sampling_rate 1 \
--work_dir "path/to/your/result/dir" --result_file "result_data.txt" 

$ python3 -m pip poetry run python3 learn_exact.py "path/to/your/result/dir"
```
