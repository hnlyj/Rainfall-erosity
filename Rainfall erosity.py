from osgeo import gdal
import numpy as np
import os
import datetime
import pandas as pd
import multiprocessing
import time
# from Mkdir import *
import traceback
import queue

alive = multiprocessing.Value('b', False)


def getBetweenDay(begin_date, end_date):
    date_list = []
    begin_date = datetime.datetime.strptime(begin_date, "%Y%m%d")
    end_date = datetime.datetime.strptime(end_date, "%Y%m%d")
    while begin_date <= end_date:
        date_str = begin_date.strftime("%Y%m%d")
        date_list.append(date_str)
        begin_date += datetime.timedelta(days=1)
    return date_list


def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + '创建成功')
    else:
        print('目录已存在')


def MKDIR(path, year, dateList):
    '''
    建立的从年到月到日的文件夹
    '''
    for i in dateList:
        path_MD = path + '\\' + year + '\\' + i[0:6] + '\\' + i
        mkdir(path_MD)


def judge(rainfall_M, Time):  # 筛选出单个点 大于0的数据 时间作为index 生成series结构
    TimeRT = []
    rainfallRT = []
    for i in range(1, len(rainfall_M)):
        if rainfall_M[i] > 0.25:
            rainfallRT.append(rainfall_M[i])
            # rainfall_M首个是索引元素 比Time list length 大 所以减个1
            TimeRT.append(Time[i - 1])
    pd_rainfall = pd.Series(rainfallRT, index=TimeRT, dtype=np.float32)
    return pd_rainfall, rainfall_M[0]


def erosionEvent(pd_rainfall, index, q):  # 处理Series结构的月数据 pd_rainfall 已剔除未降雨的时间
    Times_temp = datetime.datetime.strptime('201401010030', "%Y%m%d%H%M")
    # 单次降雨侵蚀事件经历的时间
    duration = []
    Energy = []
    # 存储单次侵蚀事件的降雨强度 筛选出最大的I
    I = []
    # 存储最大降雨强度
    I30 = []
    end = True
    # i为降雨日期 v是降雨量
    for i, v in pd_rainfall.items():
        Times_current = datetime.datetime.strptime(i, "%Y%m%d%H%M")
        delta = Times_current - Times_temp
        Times_temp = Times_current
        # 是否结束本次侵蚀的标志
        if end:
            subHour = 0
        else:
            subHour = delta.days * 24 + delta.seconds / 3600
        # 初次判断降雨侵蚀标志
        if subHour == 0:
            I.append(v)
            end = False
        # 可以进行累积的标志
        elif subHour < 6:
            I.append(v)
            end = False
        # 不能进行累积的标志 需要进行下一次 侵蚀判断
        elif subHour >= 6:
            # 清算最后一次的I数组 判断是否记录侵蚀
            I_np = np.array(I, dtype=np.float32)
            if I_np[I_np > 6.5].size != 0 and I_np.sum() < 12.7:
                # print(I_np)
                duration_single = I_np.size * 0.5
                duration.append(duration_single)
                I30.append(I_np.max())
                E = (0.29 * (1 - 0.72 * np.exp(-0.05 * 2 * I_np)))*I_np
                Energy.append(sum(E))
                # print(I)
            elif I_np.sum() > 12.7:
                duration_single = I_np.size * 0.5
                duration.append(duration_single)
                I30.append(I_np.max())
                E = (0.29 * (1 - 0.72 * np.exp(-0.05 * 2 * I_np))) * I_np
                Energy.append(sum(E))
                # print(I)
            I.clear()
            end = True
            I.append(v)
    if len(I)>0:
        I_np = np.array(I, dtype=np.float32)
        if I_np[I_np > 6.5].size != 0 and I_np.sum() < 12.7:
            # print(I_np)
            duration_single = I_np.size * 0.5
            duration.append(duration_single)
            I30.append(I_np.max())
            E = (0.29 * (1 - 0.72 * np.exp(-0.05 * 2 * I_np))) * I_np
            Energy.append(sum(E))
            # print(I)
        elif I_np.sum() > 12.7:
            duration_single = I_np.size * 0.5
            duration.append(duration_single)
            I30.append(I_np.max())
            E = (0.29 * (1 - 0.72 * np.exp(-0.05 * 2 * I_np))) * I_np
            Energy.append(sum(E))
    # 有结果的 生成DataFrame结构 节省计算资源
    if len(duration):
        single_M = pd.DataFrame({'duration': pd.Series(duration),
                                 'Energy': pd.Series(Energy),
                                 'I30': pd.Series(I30)})
        # print(single_M)
        # 计算单点的侵蚀结果
        result = np.sum(single_M['Energy'] * single_M['I30'], axis=0)
        ind_val = (index, result)  # 存索引和计算结果   注意 result是float  索引是整型 导致整型强制转化
        q.put(ind_val)


def read_M(path_D, Xsize=700, Ysize=900):
    # 1488:每个月文件夹下的文件个数 700 900：降水数据的shape
    data = np.zeros((48, Xsize, Ysize), dtype=np.float32)
    count = 0
    # 将一个月的降水数据写入矩阵
    for fileName in os.listdir(path_D):
        band_path = os.path.join(path_D, fileName)
        in_ds = gdal.Open(band_path)
        in_data = in_ds.ReadAsArray()
        # tiff中的nodata为None
        in_data[in_data == None] = np.nan
        data[count] = in_data
        # print(count)
        count += 1
    return data


# data参数是存储一个月逐半小时的降雨数据的numpy矩阵
def mainLoop(path_M, q):
    data = read_M(path_M)
    # 存储单个点一天的降雨数据/
    rainfall_M = []
    len0 = data.shape[0]
    len1 = data.shape[1]
    len2 = data.shape[2]
    data_bool = np.zeros((len1, len2), dtype=bool)
    for i in range(0, len0):
        data_bool = (data[i] > 0) | data_bool
    '''
        for k in range(0,len2):
        for j in range(0,len1):
            index=(j,k)
    '''
    indexTuple = np.where(data_bool == True)  # 一天内 亚洲地区有降水点的下标  半小时内大于0.25
    # print(indexTuple)
    lenIndex = len(indexTuple[0])
    for m in range(0, lenIndex):
        j = indexTuple[0][m]
        k = indexTuple[1][m]
        index = (j, k)
        # 首元素携带索引信息 将侵蚀结果写入对应位置
        rainfall_M.append(index)
        for i in range(0, len0):
            rainfall_M.append(data[i, j, k])
        q.put(rainfall_M[:])
        rainfall_M.clear()
    '''
    time.sleep(2)#加两秒 等待处理和写入程序完成  确保保存了数据
    '''


def WriteR(q, alive, saveEvent, Bytepath_D, DayFileName, endEvent):
    data = np.zeros((700, 900), dtype=np.float32)
    while alive.value:
        # 取索引和计算的结果 采用元组存储
        path_D=str(Bytepath_D.value,encoding='utf-8')
        try:
            ind_val = q.get(timeout=0.5)
            # print(ind_val,data.shape)
            # print(ind_val[1])
            data[ind_val[0][0], ind_val[0][1]] = ind_val[1]
        except queue.Empty:
            if saveEvent.is_set():
                saveEvent.clear()
                print(data.max())
                infoPath = os.path.join(path_D, os.listdir(path_D)[0])
                in_ds = gdal.Open(infoPath)
                in_band = in_ds.GetRasterBand(1)
                out_path = path_D.replace('AIM', 'Output')
                mkdir(out_path)
                fileNameRE = 'RE_{}.tif'.format(str(DayFileName.value,encoding='utf-8'))  # 结果文件名称
                make_raster(in_ds, os.path.join(out_path, fileNameRE), data, in_band.DataType, endEvent, nodata=np.nan)
                data = np.zeros((700, 900), dtype=np.float32)


# q1存 需要计算的数据 q2存储计算的结果
def backProcess(q1, q2, Time, alive):
    while alive.value:
        rainfall_M = q1.get()
        pd_rainfall, index = judge(rainfall_M, Time)
        erosionEvent(pd_rainfall, index, q2)


def make_raster(in_ds, fn, data, data_type, endEvent, nodata=None):
    '''
    :param in_ds: 数据指针
    :param fn: 输出路径+文件
    :param data: 结果数据矩阵
    :param data_type: 数据类型
    :param nodata: 无效值
    :return: 输出结果的指针
    '''
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(
        fn, in_ds.RasterXSize, in_ds.RasterYSize, 1, data_type
    )
    out_ds.SetProjection(in_ds.GetProjection())
    out_ds.SetGeoTransform(in_ds.GetGeoTransform())
    out_band = out_ds.GetRasterBand(1)
    if nodata is not None:
        out_band.SetNoDataValue(nodata)
    out_band.WriteArray(data)
    out_band.FlushCache()
    out_band.ComputeStatistics(False)
    endEvent.set()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    # try:

    print('*' * 40)
    print('亚洲AIMERG 一公里 30min 降雨产品之年降雨侵蚀因子计算模型')
    print('*' * 40)
    q1 = multiprocessing.Queue(maxsize=20)
    q2 = multiprocessing.Queue(maxsize=20)
    record1 = []
    block_record = []
    path = r'E:\Project Data\Rainfall\AIM'
    # path = input('请输入路径（年文件夹的上层路径）:')
    print('此机CPU {}核'.format(str(multiprocessing.cpu_count())))
    processNumber = input('请输入处理数据的子进程数：')
    processNumber = int(processNumber)
    CreateProcess = True
    saveEvent = multiprocessing.Event()
    endEvent = multiprocessing.Event()  # 确保保存影像后再进行下一次循环
    DayFileName=multiprocessing.Array('c',100)
    Bytepath_D=multiprocessing.sharedctypes.Array('c',100)
    for i in os.listdir(path):
        path_Y = os.path.join(path, i)
        for j in os.listdir(path_Y):  # 年
            path_M = os.path.join(path_Y, j)  # 月
            for k in os.listdir(path_M):  # 日
                DayFileName.value=bytes(k,encoding='utf-8')
                endEvent.clear()
                print('{}处理中，请等待！！'.format(k))
                # Writedata=np.zeros((700,900),dtype=np.float32)  #计算日降雨侵蚀因子 每日生成一个存储侵蚀结果的矩阵
                path_D = os.path.join(path_M, k)
                x=bytes(path_D,encoding='utf-8')
                Bytepath_D.value=x
                Time = []
                for fileName in os.listdir(path_D):  # 先把日期列表读取
                    Time.append(fileName[7:-4])
                front_func = multiprocessing.Process(target=mainLoop, args=(path_D, q1))  # 前台函数
                front_func.start()
                alive.value = True
                if CreateProcess:
                    for i in range(0, processNumber):
                        back_func = multiprocessing.Process(target=backProcess, args=(q1, q2, Time, alive))  # 后台任务处理队列
                        back_func.start()
                        block_record.append(back_func)
                    write_func = multiprocessing.Process(target=WriteR,
                                                         args=(q2, alive, saveEvent, Bytepath_D, DayFileName, endEvent))
                    write_func.start()
                    block_record.append(write_func)
                    CreateProcess = False
                front_func.join()
                while True:  # 监控处理数据的进程 结束后再进行下一次的循环
                    if q1.empty() and q2.empty():  # 前台函数处理完毕后 ，存放数据的队列为空的话  杀死处理数据的子进程
                        time.sleep(0.5)
                        saveEvent.set()
                        break
                # 保存处理的结果
                print('{}已经处理完毕！'.format(k))
                endEvent.wait()
    # except Exception as e:
    #     with open('.\\error.txt','a') as f:
    #         traceback.print_exc(file=f)
