import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

args = sys.argv
# 第一引き数は出力ファイルパス
# 第2引き数は数字の最大値

def main():
    file_path = args[1];
    for i in range(int(args[2])+1):
        data_csv = file_path + f"/original_{i}.csv"
        data = pd.read_csv(data_csv, header=None)
        # data = pd.read_csv(file_path, header=None)
        for index in range(data.shape[0]):
                    
            pixels = data.iloc[index].values
            image = pixels.reshape(28, 28)

            # 画像の表示
            plt.imshow(image, cmap='gray')
            plt.title(f' label:{i} original')
            plt.axis('off')
            plt.savefig(f"{file_path}/png/{i}_original.png", bbox_inches='tight', pad_inches=0)
            plt.close()

    for i in range(int(args[2])+1):
        data_csv = file_path + f"/learned_{i}.csv"
        data = pd.read_csv(data_csv, header=None)
        # data = pd.read_csv(file_path, header=None)
        j = 1 
        for index in range(data.shape[0]):
                    
            # data_csv = file_path + f"/learned_{i}.csv"
            # data = pd.read_csv(data_csv, header=None)
            # pixels = row.values
            pixels = data.iloc[index].values
            # pixels = data.values

            # 28x28の画像に変換
            image = pixels.reshape(28, 28)

            # 画像の表示
            plt.imshow(image, cmap='gray')
            plt.title(f' label:{i} epoch:{j}')
            plt.axis('off')
            plt.savefig(f"{file_path}/png/{i}_{j}epoch.png", bbox_inches='tight', pad_inches=0)
            plt.close()
            j += 1

if __name__=="__main__": 
    main();
