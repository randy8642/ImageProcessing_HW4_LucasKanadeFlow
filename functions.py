import cv2
import numpy as np
from collections import deque

def LK_opticalFlow(img_prev, img_next, trackingPoint, window_size=[15, 15]):
    img_prev_gray = cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY).astype(np.float32)
    img_next_gray = cv2.cvtColor(img_next, cv2.COLOR_BGR2GRAY).astype(np.float32)

    Iy = (img_prev_gray[2:, 1:-1] - img_prev_gray[1:-1, 1:-1]) / 2
    Ix = (img_prev_gray[1:-1, 2:] - img_prev_gray[1:-1, 1:-1]) / 2
    # It = img_next_gray[1:-1, 1:-1] - img_prev_gray[1:-1, 1:-1]
    
    iter_points = []
    for (Py, Px) in trackingPoint:
        iter_point = [[Py, Px]]
        n = 0
        pre_v = np.inf
        
        while True:  
            n += 1  
            crop_x_upper = int(Px + window_size[1] // 2)
            crop_x_lower = int(Px - window_size[1] // 2 - 1)
            crop_y_upper = int(Py + window_size[1] // 2)
            crop_y_lower = int(Py - window_size[1] // 2 - 1)

            if n == 1:
                crop_prev_gray = img_prev_gray[crop_y_lower:crop_y_upper, crop_x_lower:crop_x_upper]
            crop_next_gray = img_next_gray[crop_y_lower:crop_y_upper, crop_x_lower:crop_x_upper]

            sigma = 0.3 * ((window_size[0] - 1) * 0.5 - 1) + 0.8
            guass_kernal = get_guassKernal(l=window_size[0], sig=sigma)

            sub_Iy = Iy[crop_y_lower:crop_y_upper, crop_x_lower:crop_x_upper] * guass_kernal
            sub_Ix = Ix[crop_y_lower:crop_y_upper, crop_x_lower:crop_x_upper] * guass_kernal
            sub_It = (crop_next_gray - crop_prev_gray) * guass_kernal
            
            sub_Iy = sub_Iy.flatten()
            sub_Ix = sub_Ix.flatten()
            sub_It = sub_It.flatten()

            A = np.vstack([sub_Ix, sub_Iy]).T
            b = sub_It * (-1.)
            
            v = np.linalg.pinv(A.T @ A) @ A.T @ b
            
            v_abs = np.sqrt(v[0]**2 + v[1]**2)

            if (v_abs > pre_v*3):                
                break
            if (v_abs < 1.2):                
                break

            Px, Py = (np.array([Px, Py]) + np.round(v)).astype(np.int32)
            
            iter_point.append([Py, Px])

            if n > 70:                
                break
            
            pre_v = v_abs

        iter_point = np.array(iter_point, dtype=np.int32)
        iter_points.append(iter_point)
    
    return iter_points

def get_guassKernal(l=5, sig=1.) -> np.ndarray:
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def LK_opticalFlow_openCV(img_prev, img_next, trackingPoint, window_size=[15, 15]):
    img_prev_gray = cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY)
    img_next_gray = cv2.cvtColor(img_next, cv2.COLOR_BGR2GRAY)
    

    iter_points = []
    for (Py, Px) in trackingPoint:
        
        p0 = np.array([Py, Px])
        p0 = np.expand_dims(p0, axis=0)
        p0 = np.expand_dims(p0, axis=0)
        p0 = p0.astype(np.float32)
        
        lk_params = {
            "winSize": window_size,
            "maxLevel": 1,
            "criteria": (cv2.TERM_CRITERIA_COUNT, 1, 0),
            'flags': cv2.OPTFLOW_USE_INITIAL_FLOW
        }

        iter_point = [p0]
        
        pre_err = 0
        while True:    
            p1, st, err = cv2.calcOpticalFlowPyrLK(img_prev_gray, img_next_gray, iter_point[0].copy(), iter_point[-1].copy(), **lk_params)
            iter_point.append(p1)           
            print(err)
            if abs(err - pre_err) < 0.03:
                break
            pre_err = err

        iter_point = np.array(iter_point, dtype=np.int32)
        iter_point = iter_point[:, 0, 0, :]
        
        iter_points.append(iter_point)
    
    return iter_points