# 1. 베이스 이미지 (CUDA 12.1, cuDNN 8, Ubuntu 22.04)
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# 2. 환경 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# 3. Python 및 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-venv \
    python3-tk fonts-nanum \
    && rm -rf /var/lib/apt/lists/*

# 4. 작업 디렉토리 설정
WORKDIR /workspace

# 5. 가상 환경 생성 및 설정
ENV VIRTUAL_ENV=/opt/venv/py310
RUN python3.10 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# ... (이전 내용은 동일) ...

# 6. pip 업그레이드 및 requirements.txt 복사/설치
RUN pip install --upgrade pip
COPY requirements.txt .
# 라이브러리 설치
RUN pip install --no-cache-dir -r requirements.txt

# [추가됨] 가상환경을 Jupyter Kernel로 자동 등록
# 이렇게 하면 VS Code에서 별도 설정 없이 바로 이 커널을 선택할 수 있습니다.
RUN python -m ipykernel install --user --name=docker_env --display-name "Python (Docker Venv)" # 이름등록하는 코드,이름등록해야 쉽게 찾을 수 있음
# 서버 여러개일때 선택쉬움

# 7. JupyterLab 실행 명령
# ... (이후 내용은 동일) ...
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser", "--ServerApp.token=''", "--ServerApp.password=''"]