import subprocess
import psutil
import pynvml  #导包


def get_free_gpu_memory():
    try:
        _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

        ACCEPTABLE_AVAILABLE_MEMORY = 1024
        COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = _output_to_list(
            subprocess.check_output(COMMAND.split()))[1:]
        memory_free_values = [
            int(x.split()[0]) for i, x in enumerate(memory_free_info)
        ]
        return memory_free_values
    except Exception as e:
        print("Could not print GPU memory info: ", e)
        return []


def get_gpu_info():

    UNIT = 1024 * 1024

    pynvml.nvmlInit()  #初始化

    gpuDeviceCount = pynvml.nvmlDeviceGetCount()  #获取Nvidia GPU块数
    print("GPU个数：", gpuDeviceCount)

    for i in range(gpuDeviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(
            i)  #获取GPU i的handle，后续通过handle来处理

        memoryInfo = pynvml.nvmlDeviceGetMemoryInfo(handle)  #通过handle获取GPU i的信息

        print("第 %d 张卡：" % i, "-" * 30)
        print("剩余容量：", memoryInfo.free / UNIT, "MB")
        """
        # 设置显卡工作模式
        # 设置完显卡驱动模式后，需要重启才能生效
        # 0 为 WDDM模式，1为TCC 模式
        gpuMode = 0     # WDDM
        gpuMode = 1     # TCC
        pynvml.nvmlDeviceSetDriverModel(handle, gpuMode)
        # 很多显卡不支持设置模式，会报错
        # pynvml.nvml.NVMLError_NotSupported: Not Supported
        """
    pynvml.nvmlShutdown()  #最后关闭管理工具


if __name__ == '__main__':
    print(get_free_gpu_memory())
    print(get_gpu_info())
