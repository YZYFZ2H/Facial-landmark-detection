import dlib
import cv2
import numpy as np
import time

# 加载人脸特征点检测器
predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# 打开摄像头
cam = cv2.VideoCapture(0)
cam.set(3, 1280)  # 设置摄像头宽度
cam.set(4, 720)  # 设置摄像头高度

# 设置绘图参数
color_white = (255, 255, 255)  # 绘图颜色
line_width = 3  # 绘图线宽

# 设置光流法参数
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# 初始化跟踪器列表和位置列表
tracker_list = []
pos_list = []

# 设置帧率和计数器
frame_rate = 30
counter = 0

# 获取计时器频率
tick_frequency = cv2.getTickFrequency()

# 开始循环
while True:
    start_tick = cv2.getTickCount()  # 获取当前计时器的值
    ret_val, img = cam.read()  # 读取一帧图像
    if img is None:  # 判断输入图像是否为空
        continue  # 如果为空，跳过本次循环
    if counter % frame_rate == 0:  # 每隔一定时间检测一次人脸
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换颜色空间
        dets = detector(rgb_image)  # 检测人脸位置
        tracker_list = []  # 清空跟踪器列表
        for det in dets:  # 对于每个检测到的人脸
            tracker = dlib.correlation_tracker()  # 创建跟踪器
            tracker.start_track(rgb_image, det)  # 开始跟踪
            tracker_list.append(tracker)  # 将跟踪器添加到列表中
            cv2.rectangle(img, (det.left(), det.top()), (det.right(), det.bottom()), color_white, line_width)  # 绘制人脸矩形框
            shape = predictor(rgb_image, det)  # 检测人脸特征点
            for i in range(68):  # 绘制人脸特征点
                cv2.circle(img, (shape.part(i).x, shape.part(i).y), 2, (0, 255, 0), -1)
        if len(pos_list) == 0:  # 如果位置列表为空，则初始化为整个图像大小
            pos_list = [dlib.rectangle(0, 0, img.shape[1], img.shape[0])] * len(tracker_list)
    else:  # 否则执行光流法跟踪
        for i, tracker in enumerate(tracker_list):
            ret_val, img = cam.read()  # 读取一帧图像
            if img is None:  # 判断输入图像是否为空
                continue  # 如果为空，跳过本次循环
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换颜色空间

            # 检查 prev_image 和 pos 是否为空
            if 'prev_image' not in locals() or prev_image is None:
                prev_image = rgb_image.copy()
            if len(pos_list) == 0 or pos_list[i] is None:
                pos_list[i] = dlib.rectangle(0, 0, img.shape[1], img.shape[0])

            prev_pos = tracker.get_position()  # 获取上一帧跟踪器的位置
            prev_pts = [[prev_pos.left(), prev_pos.top()], [prev_pos.right(), prev_pos.bottom()]]
            prev_pts = np.array(prev_pts, dtype=np.float32).reshape(-1, 1, 2)  # 将位置转换为数组形式
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_image, rgb_image, prev_pts, None, **lk_params)  # 计算光流
            good_next_pts = next_pts[status == 1]  # 选择有效的光流点
            if len(good_next_pts) > 0:  # 如果有有效的光流点
                x_min, y_min = np.min(good_next_pts, axis=0).astype(np.int32)  # 计算新位置的左上角坐标
                x_max, y_max = np.max(good_next_pts, axis=0).astype(np.int32)  # 计算新位置的右下角坐标
                tracker.update(rgb_image, dlib.rectangle(x_min, y_min, x_max, y_max))  # 更新跟踪器

                # 修正 pos 的计算
                pos_list[i] = tracker.get_position()  # 获取跟踪器的新位置
                if pos_list[i] is not None:  # 如果新位置不为空
                    left = int(pos_list[i].left())  # 获取新位置的左上角坐标
                    top = int(pos_list[i].top())
                    right = int(pos_list[i].right())  # 获取新位置的右下角坐标
                    bottom = int(pos_list[i].bottom())

                    # 检查 left、top、right、bottom 是否超出图像边界
                    if left < 0:
                        left = 0
                    if top < 0:
                        top = 0
                    if right > img.shape[1]:
                        right = img.shape[1]
                    if bottom > img.shape[0]:
                        bottom = img.shape[0]

                    try:
                        cv2.rectangle(img, (left, top), (right, bottom), color_white, line_width)  # 绘制人脸矩形框
                    except cv2.error as e:
                        print(f"Error: {e}")
                    roi = dlib.rectangle(left, top, right, bottom)  # 获取新位置的矩形框
                    shape = predictor(rgb_image, roi)  # 检测人脸特征点
                    for j in range(68):  # 绘制人脸特征点
                        cv2.circle(img, (shape.part(j).x, shape.part(j).y), 2, (0, 255, 0),-1)
            else:  # 如果没有有效的光流点，将位置设置为 None
                pos_list[i] = None

            prev_image = rgb_image.copy()  # 将当前图像设置为上一帧图像

    counter += 1  # 计数器加一
    end_tick = cv2.getTickCount()  # 获取当前计时器的值
    tick_diff = end_tick - start_tick  # 计算计时器的差值
    time_diff = tick_diff / tick_frequency  # 计算时间差值
    sleep_ms = int(max(1, int(1000 / frame_rate - time_diff * 1000)))  # 计算休眠时间
    time.sleep(sleep_ms / 1000)  # 休眠一定时间

    cv2.imshow('Face Detection', img)  # 显示图像
    if cv2.waitKey(1) == 27:  # 如果按下 ESC 键，退出程序
        break
