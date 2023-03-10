from torch import cuda

cuda.is_available()
cuda.device_count() > 0
print(cuda.get_device_name(cuda.current_device()))