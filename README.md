# ImageProcessing_HW4_LucasKanadeFlow
 
# 作業說明
## 功能
實現 Lucas-Kanade Flow 追蹤方法，並在兩組影像(Cup0.Jpg, Cup1.Jpg與Pillow0.Jpg, Pillow1.Jpg)上，隨意指定兩個以上的特徵點去做追蹤。
## 需求
1. 在第一張影像上(Cup0.Jpg與Pillow0.Jpg)標記出選取的特徵點並輸出成影像。如下圖所示(藍色的兩個點是選取的特徵點):  
![作業說明1](/img/作業說明1.png)
2. 在第二張影像上(Cup1.Jpg與Pillow1.Jpg)將 Lucas-Kanade Flow 每一個 iteration 的位置與最後追蹤收斂的位置標記出來並輸出成影像，如下圖所示:  
![作業說明2](/img/作業說明2.png)  
紅色點是中間 iteration 的點，黃綠色的點是最後追蹤到的位置，點與點之間用紫色的線連接。

# 環境
- python v3.9 [網站](https://pipenv-fork.readthedocs.io/en/latest/)
- pipenv 套件管理工具 [網站](https://pipenv-fork.readthedocs.io/en/latest/) 

# 使用說明
1. 下載專案
2. 移至專案目錄\
`cd /d ImageProcessing_HW4_LucasKanadeFlow`
2. 安裝所需套件\
`pipenv install`

# 功能實現
## Lucas-Kanade Optical Flow
### 程式碼
```python
def LK_opticalFlow(img_prev, img_next, trackingPoint, window_size=[15, 15]):
    img_prev_gray = cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY).astype(np.float32)
    img_next_gray = cv2.cvtColor(img_next, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # 計算x方向梯度/y方向梯度/時間變化的差異
    Iy = (img_prev_gray[2:, 1:-1] - img_prev_gray[1:-1, 1:-1]) / 2
    Ix = (img_prev_gray[1:-1, 2:] - img_prev_gray[1:-1, 1:-1]) / 2
    It = img_next_gray[1:-1, 1:-1] - img_prev_gray[1:-1, 1:-1]
    
    iter_points = []
    # 每個點分開執行追蹤
    for (Py, Px) in trackingPoint:
        iter_point = [[Py, Px]]
        n = 0
        pre_v = np.inf
        while True:  
            n += 1
            # 計算要裁切的上下界
            crop_x_upper = int(Px + window_size[1] // 2)
            crop_x_lower = int(Px - window_size[1] // 2 - 1)
            crop_y_upper = int(Py + window_size[1] // 2)
            crop_y_lower = int(Py - window_size[1] // 2 - 1)

            mask = np.zeros(It.shape, dtype=np.bool8)
            mask[crop_y_lower:crop_y_upper, crop_x_lower:crop_x_upper] = True

            # 產生高斯分布的權重
            sigma = 0.3 * ((window_size[0] - 1) * 0.5 - 1) + 0.8
            guass_kernal = get_guassKernal(l=window_size[0], sig=sigma).flatten()

            # 取得依照權重重新分配後的梯度數值
            sub_Iy = Iy[mask] * guass_kernal
            sub_Ix = Ix[mask] * guass_kernal
            sub_It = It[mask] * guass_kernal

            # 計算x方向與y方向的移動量
            A = np.vstack([sub_Ix, sub_Iy]).T
            b = -sub_It
            
            v = np.linalg.pinv(A.T @ A) @ A.T @ b

            # 判斷是否停止
            v_abs = np.sqrt(v[0]**2 + v[1]**2)

            if (v_abs > pre_v*2):                
                break
            if (v_abs < 1):                
                break
            
            # 更新位置
            Px, Py = (np.array([Px, Py]) + np.round(v)).astype(np.int32)
            
            # 紀錄迭代的過程
            iter_point.append([Py, Px])

            if n > 70:                
                break
            
            pre_v = v_abs

        iter_point = np.array(iter_point, dtype=np.int32)
        iter_points.append(iter_point)
    
    return iter_points
```

```python
# 產生高斯核
def get_guassKernal(l=5, sig=1.) -> np.ndarray:
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)
```
### 說明
根據Lucas-Kanade Optical Flow公式計算光流以追蹤特徵點。該方法假設兩個相鄰的影格的圖像內容位移很小，且位移在目標點(P)的鄰域內為大致為常數。假設光流方程式對於以P點為中心的window內的所有像素都成立。  
![](/img/LK說明01.jpg)  
- `q`為像素點
- `Ix`為對x方向的梯度
- `Iy`為對y方向的梯度
- `It`為對時間的梯度

上述式子寫成矩陣形式即為`Av=b`  
![](/img/LK說明02.jpg)  
可推得
![](/img/LK說明03.jpg)  
因所使用的梯度均為1次導數因此需透過迭代的方式來求得最佳解。  
![](/img/Optical-flow-estimation-Left-the-Lucas-Kanade-Right-the-Lucas-Kanade-aided-by.png)  
[圖片來源](https://www.researchgate.net/figure/Optical-flow-estimation-Left-the-Lucas-Kanade-Right-the-Lucas-Kanade-aided-by_fig1_280567385)

# 介面操作說明
## Step 1：開啟程式並選擇第一張圖片
![介面說明01](/img/介面說明01.jpg)
## Step 2：選擇第二張圖片
![介面說明02](/img/介面說明02.jpg)
## Step 3：在第一張圖片上點選
使用滑鼠在第一張圖片上點選要追蹤的點  
![介面說明03](/img/介面說明03.jpg)
## Step 4：按下開始
![介面說明04](/img/介面說明04.jpg)
## Step 5：完成
可按下`Save`儲存結果圖片  
![介面說明05](/img/介面說明05.jpg)

# 結果
- `綠色`為原選取的座標  
- `藍色`為追蹤結果  
- `紅色`為迭代的過程點  
- `紫色線段`為移動過程  
## 樣張1
![](/results/Cup0.jpg)
## 樣張2
![](/results/Pillow0.jpg)