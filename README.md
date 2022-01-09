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

# 功能實現

# 處理步驟