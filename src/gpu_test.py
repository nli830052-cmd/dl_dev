# image만 있으면 언제든지 contina는 실행가능함
# 항상주의
# container up / down -> 만들고 지우고
# container start / stop -> 실행하고 정지
# GPU 썻을때와 안썼을때의 차이 비교 코드
# 나중에 음성인식에서는 numpy 1.26버전 사용해야함

import torch
import time

# 1. 장치 설정: 코드를 장치 독립적으로 만드는 핵심 패턴
# torch.cuda.is_available()을 확인하여 GPU 사용 가능 여부를 판단합니다.
# 가능하면 'cuda' 장치를, 불가능하면 'cpu' 장치를 사용하도록 설정합니다.
# 이렇게 하면 코드가 어떤 환경에서도 유연하게 동작할 수 있습니다.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")


# 2. 텐서를 특정 장치로 생성하거나 이동
# 처음부터 GPU에 텐서 생성
tensor_on_gpu = torch.randn(1000, 1000, device=device)
print(f"\n텐서가 생성된 장치: {tensor_on_gpu.device}")


# CPU에서 생성 후 GPU로 이동
tensor_on_cpu = torch.randn(1000, 1000)
print(f"이동 전 텐서 장치: {tensor_on_cpu.device}")
tensor_on_gpu_moved = tensor_on_cpu.to(device)
print(f"이동 후 텐서 장치: {tensor_on_gpu_moved.device}")


# 3. GPU 연산 속도 비교
# 큰 행렬 곱셈 연산을 CPU와 GPU에서 각각 수행하고 시간을 측정합니다.
size = 4096
cpu_a = torch.randn(size, size, device='cpu')
cpu_b = torch.randn(size, size, device='cpu')


# CPU 연산 시간 측정
start_time_cpu = time.time()
result_cpu = torch.matmul(cpu_a, cpu_b)
end_time_cpu = time.time()
cpu_time = end_time_cpu - start_time_cpu
print(f"\nCPU 연산 시간: {cpu_time:.5f} 초")


# GPU 연산 시간 측정
if device.type == 'cuda':
    gpu_a = cpu_a.to(device)
    gpu_b = cpu_b.to(device)


    # GPU 워밍업 (초기 커널 로딩 시간 제외)
    _ = torch.matmul(gpu_a, gpu_b)
   
    # GPU 연산은 비동기적으로 처리될 수 있으므로, 정확한 시간 측정을 위해
    # torch.cuda.synchronize()를 호출하여 GPU의 모든 연산이 끝날 때까지 기다립니다.
    torch.cuda.synchronize()
    start_time_gpu = time.time()
    result_gpu = torch.matmul(gpu_a, gpu_b)
    torch.cuda.synchronize()
    end_time_gpu = time.time()
    gpu_time = end_time_gpu - start_time_gpu
    print(f"GPU 연산 시간: {gpu_time:.5f} 초")
   
    # 속도 향상률 계산
    speedup = cpu_time / gpu_time
    print(f"GPU가 CPU보다 약 {speedup:.2f}배 빠릅니다.")

