import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import random
import copy
import warnings
warnings.simplefilter('ignore')

#メイン
def main():
    #初期解生成
    dna_1 = set_param()
    dna_2 = set_param()
    dna_3 = set_param()
    counter = 0
    for sedai in range(10000):
        #交叉
        dna_4, dna_5, dna_6, dna_7, dna_8, dna_9 = closs(dna_1, dna_2, dna_3)
        #突然変異
        dna_4, dna_5, dna_6, dna_7, dna_8, dna_9 = mutation(dna_4, dna_5, dna_6, dna_7, dna_8, dna_9)
        #評価
        eval_lst = evalate(dna_1, dna_2, dna_3, dna_4, dna_5, dna_6, dna_7, dna_8, dna_9)
        #出力
        output(dna_1, dna_2, dna_3, dna_4, dna_5, dna_6, dna_7, dna_8, dna_9, eval_lst, sedai)
        #終了条件
        if np.amax(eval_lst) == 1.0:
            break
        #淘汰
        dna_1, dna_2, dna_3 = selection(dna_1, dna_2, dna_3, dna_4, dna_5, dna_6, dna_7, dna_8, dna_9, eval_lst)


#出力
def output(dna_1, dna_2, dna_3, dna_4, dna_5, dna_6, dna_7, dna_8, dna_9, eval_lst, sedai):
    dna_lst = [dna_1, dna_2, dna_3, dna_4, dna_5, dna_6, dna_7, dna_8, dna_9]
    max_idx = np.argmax(eval_lst)
    print('第{}世代'.format(sedai))
    print(eval_lst)
    print(dna_lst)
    print(dna_lst[max_idx])
    print('Test accuracy: {}'.format(eval_lst[max_idx]))
    print()


#パラメータを設定
def set_param():
    # 配列のランダム抽出
    l1 = random.choice([8, 10, 15, 20, 25, 30, 35, 40, 45, 50])#, 60, 80, 100])
    l2 = random.choice([8, 10, 15, 20, 25, 30, 35, 40, 45, 50])#, 60, 80, 100])
    l1_drop = random.choice([0.0, 0.3])
    l2_drop = random.choice([0.0, 0.3])
    epochs = random.choice([100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500])
    batch_size = random.choice([8, 16, 32, 64])
    return [l1, l2, l1_drop, l2_drop, epochs, batch_size]


#交叉
def closs(dna_1, dna_2, dna_3):
    dna_4 = copy.copy(dna_1)
    dna_5 = copy.copy(dna_1)
    dna_6 = copy.copy(dna_2)
    dna_7 = copy.copy(dna_2)
    dna_8 = copy.copy(dna_3)
    dna_9 = copy.copy(dna_3)
    #一様交叉
    Mask = [random.choice([0, 1]) for i in range(len(dna_1))]
    for i, mask in enumerate(Mask):
        if mask == 0:
            dna_5[i] = dna_2[i]
        else:
            dna_4[i] = dna_2[i]
    Mask = [random.choice([0, 1]) for i in range(len(dna_1))]
    for i, mask in enumerate(Mask):
        if mask == 0:
            dna_7[i] = dna_3[i]
        else:
            dna_6[i] = dna_3[i]
    Mask = [random.choice([0, 1]) for i in range(len(dna_1))]
    for i, mask in enumerate(Mask):
        if mask == 0:
            dna_8[i] = dna_1[i]
        else:
            dna_9[i] = dna_1[i]
    return dna_4, dna_5, dna_6, dna_7, dna_8, dna_9


#突然変異
def mutation(dna_4, dna_5, dna_6, dna_7, dna_8, dna_9):
    dna_lst = [dna_4, dna_5, dna_6, dna_7, dna_8, dna_9]
    for i, dna in enumerate(dna_lst):
        #30%の確率で突然変異
        x = True if 45 >= random.randint(0,100) else False
        if x == True:
            rand_num = random.choice([1,2,3,4])
            rand_posi = random.sample(range(len(dna)), k=rand_num)
            mask_dna = set_param()
            for posi in rand_posi:
                dna[posi] = mask_dna[posi]
    return dna_4, dna_5, dna_6, dna_7, dna_8, dna_9


#評価
def evalate(dna_1, dna_2, dna_3, dna_4, dna_5, dna_6, dna_7, dna_8, dna_9):
    #全データを読み取る, csvファイルからPandas DataFrameへ読み込み
    data = pd.read_csv('train_3.csv', header=None, delimiter=',', low_memory=False)

    dna_lst = [dna_1, dna_2, dna_3, dna_4, dna_5, dna_6, dna_7, dna_8, dna_9]
    train_scope = [(251, 1000), (501, 250), (751, 500), (1, 750)]  #データを分割するための範囲
    test_scope = [(1, 250), (251, 500), (501, 750), (751, 1000)]  #データを分割するための範囲
    eval_lst = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  #評価を格納する配列
    for count, dna in enumerate(dna_lst):
        for i in range(4):
            #訓練データとテストデータを設定
            if i != 3 or i != 0:
                train = data[(data[7] >= train_scope[i][0]) | (data[7] <= train_scope[i][1])]
                test = data[(data[7] >= test_scope[i][0]) & (data[7] <= test_scope[i][1])]
            else:
                train = data[(data[7] >= train_scope[i][0]) & (data[7] <= train_scope[i][1])]
                test = data[(data[7] >= test_scope[i][0]) & (data[7] <= test_scope[i][1])]

            #trainのtargetをカテゴリーに変換
            train[6] = train[6].astype('category')
            test[6] = test[6].astype('category')

            # ラベルエンコーディング（LabelEncoder）
            #訓練データ
            le = LabelEncoder()
            encoded = le.fit_transform(train[6].values)
            decoded = le.inverse_transform(encoded)
            train[6] = encoded
            #テストデータ
            le = LabelEncoder()
            encoded = le.fit_transform(test[6].values)
            decoded = le.inverse_transform(encoded)
            test[6] = encoded

            #データとラベルを分割する
            x_train, y_train = train.drop([6], axis=1).drop([7], axis=1), train[6]
            x_test, y_test = test.drop([6], axis=1).drop([7], axis=1), test[6]

            #モデルを構築
            model = keras.Sequential([
                keras.layers.Flatten(input_shape=(6,)),
                keras.layers.Dense(dna[0], activation='relu'),
                keras.layers.Dropout(dna[2]),
                keras.layers.Dense(dna[1], activation='relu'),
                keras.layers.Dropout(dna[3]),
                keras.layers.Dense(5, activation='softmax')
            ])

            #モデルをコンパイル
            model.compile(optimizer='adam', 
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])

            #訓練データを適用
            model.fit(x_train, y_train, epochs=dna[4], batch_size=dna[5], verbose=0)

            #テストデータを適用
            test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

            #評価を格納
            eval_lst[count] += test_acc

    #評価値を平均値にする
    eval_lst = eval_lst / 4
    return eval_lst


#淘汰
def selection(dna_1, dna_2, dna_3, dna_4, dna_5, dna_6, dna_7, dna_8, dna_9, eval_lst):
    dna_lst = [dna_1, dna_2, dna_3, dna_4, dna_5, dna_6, dna_7, dna_8, dna_9]
    #トーナメント形式で次世代を決定する
    next_gene = []
    tnmt_num_list = random.sample(range(len(dna_lst)), k=len(dna_lst))
    #一回戦
    eval_tnm_lst = [eval_lst[tnmt_num_list[0]], eval_lst[tnmt_num_list[1]], eval_lst[tnmt_num_list[2]]]
    next_gene.append(tnmt_num_list[np.argmax(eval_tnm_lst)])
    #二回戦
    eval_tnm_lst = [eval_lst[tnmt_num_list[3]], eval_lst[tnmt_num_list[4]], eval_lst[tnmt_num_list[5]]]
    next_gene.append(tnmt_num_list[np.argmax(eval_tnm_lst)+3])
    #三回戦
    eval_tnm_lst = [eval_lst[tnmt_num_list[6]], eval_lst[tnmt_num_list[6]], eval_lst[tnmt_num_list[8]]]
    next_gene.append(tnmt_num_list[np.argmax(eval_tnm_lst)+6])
    return dna_lst[next_gene[0]], dna_lst[next_gene[1]], dna_lst[next_gene[2]]


if __name__ == '__main__':
    main()