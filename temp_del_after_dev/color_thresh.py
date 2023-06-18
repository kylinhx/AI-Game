import cv2
import numpy as np

def get_point(event, x, y, flags, param):
    # 鼠标单击事件
    if event == cv2.EVENT_LBUTTONDOWN:
        # 输出坐标
        print('坐标值: ', x, y)
        # 在传入参数图像上画出该点
        #cv2.circle(param, (x, y), 1, (255, 255, 255), thickness=-1)
        img = param.copy()
        # 输出坐标点的像素值
        print('像素值：',param[y][x]) # 注意此处反转，(纵，横，通道)
        # 显示坐标与像素
        text = "("+str(x)+','+str(y)+')'+str(param[y][x])
        cv2.putText(img,text,(0,param.shape[0]),cv2.FONT_HERSHEY_PLAIN,1.5,(0,0,255),1)
        cv2.imshow('image', img)
        cv2.waitKey(0)

# 根据截取的blood，返回blood情况
def handle_blood_(blood_image):
    # 判断绿色通道灰度值计算血量
    image_g = blood_image[:,:,1]
    threshold_value = 215 # 阈值可以根据应用程序的需求进行调整
    ret, bw_img = cv2.threshold(image_g, threshold_value, 255, cv2.THRESH_TOZERO)

    # bw_img是二值化图像
    total_pixels = bw_img.shape[0] * bw_img.shape[1] # 计算图像中所有像素的数量
    num_white_pixels = np.count_nonzero(bw_img) # 计算像素值不为0的点的数量
    white_ratio = num_white_pixels / total_pixels # 计算像素值不为0的点占所有点的比率
    
    # white_ratio 即可看作是百分比血量
    return white_ratio, bw_img

# 根据截取的defend，返回defend的情况
def handle_defend_(defend_image):
    # 判断蓝色通道灰度值计算血量
    image_b = defend_image[:,:,0]
    threshold_value = 150 # 阈值可以根据应用程序的需求进行调整
    ret, bw_img = cv2.threshold(image_b, threshold_value, 255, cv2.THRESH_TOZERO)

    # bw_img是二值化图像
    total_pixels = bw_img.shape[0] * bw_img.shape[1] # 计算图像中所有像素的数量
    num_white_pixels = np.count_nonzero(bw_img) # 计算像素值不为0的点的数量
    white_ratio = num_white_pixels / total_pixels # 计算像素值不为0的点占所有点的比率
    
    # white_ratio 即可看作是百分比血量
    return white_ratio, bw_img

# 根据截取的energy，返回energy的情况
def handle_energy_(energy_image):
    # 判断蓝色通道灰度值计算血量
    image_r = energy_image[:,:,2]
    threshold_value = 200 # 阈值可以根据应用程序的需求进行调整
    ret, bw_img = cv2.threshold(image_r, threshold_value, 255, cv2.THRESH_TOZERO)

    # bw_img是二值化图像
    total_pixels = bw_img.shape[0] * bw_img.shape[1] # 计算图像中所有像素的数量
    num_white_pixels = np.count_nonzero(bw_img) # 计算像素值不为0的点的数量
    white_ratio = num_white_pixels / total_pixels # 计算像素值不为0的点占所有点的比率
    
    # white_ratio 即可看作是百分比血量
    return white_ratio, bw_img

if __name__ == "__main__":
    image = cv2.imread(r'temp_del_after_dev\energy_img.png')
    # image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    # 定义窗口 并绑定事件 传入各自对应的参数
    cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback('image', get_point, image)

    # 显示图像
    while(True):
        cv2.imshow('image', image)
        if cv2.waitKey(20) & 0xFF == 27:
            break

    # print(handle_blood_(image)[0])
    print(handle_energy_(image)[0])
    cv2.imshow('', handle_energy_(image)[1])
    cv2.waitKey(0)