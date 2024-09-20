# learn_mnist
2値化されたMnistの学習コード\
1枚の画像を1行にカンマ区切りされた状態で保存したcsvファイルを使用

```
$ ./learn_mnist --v_dim 784 --h_dim 1000 --nabla 0.01 --epoch 100  --batch_size 200 --sampling_rate 1 \ 
--data_file "path/to/your/mnist/data/file.csv"  --work_dir "path/to/your/result/dir"
$ mkdir "path/to/your/result/dir/png"
$ python3 -m pip poetry run python3 data2png.py "path/to/your/result/dir" 9
```
