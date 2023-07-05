import numpy as np

np.random.seed(seed=1234)

def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    ''' parameters:
    p : probability of random erasing
    s_l : erasing ratio lower bound
    s_h : erasing ratio upper bound
    r_1 : erasing aspects ratio lower bound
    r_2 : erasing aspects ratio upper bound
    v_l : lower bound pixel value for the erased area
    v_h : upper bound pixel value for the erased area
    pixel_level : pixel-level randomization for erased area
    
    Random Erasing Data Augmentation
    Zhun Zhong, Liang Zheng, Guoliang Kang, Shaozi Li, Yi Yang
    https://arxiv.org/abs/1708.04896
    '''
    def eraser(input_img):
        if input_img.ndim == 3:
            img_h, img_w, img_c = input_img.shape
        elif input_img.ndim == 2:
            img_h, img_w = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img
        s = img_h*img_w
        while True:
            s_e = np.random.uniform(s_l, s_h) * s
            r_e = np.random.uniform(r_1, r_2)
            h_e = int(np.sqrt(s_e * r_e))
            w_e = int(np.sqrt(s_e / r_e))
            
            x_e = np.random.randint(0, img_w)
            y_e = np.random.randint(0, img_h)

            if x_e + w_e <= img_w and y_e + h_e <= img_h:
                break

        if pixel_level:
            if input_img.ndim == 3:
                c = np.random.uniform(v_l, v_h, (h_e, w_e, img_c))
            if input_img.ndim == 2:
                c = np.random.uniform(v_l, v_h, (h_e, w_e))
        else:
            c = np.random.uniform(v_l, v_h)
        # Area[y_e:y_e + h_e, x_e:x_e + w_e] got erased/replaced by c
        input_img[y_e:y_e + h_e, x_e:x_e + w_e] = c 
        return input_img
    return eraser
