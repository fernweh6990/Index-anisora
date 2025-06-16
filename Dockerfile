FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04 as base

# 시스템 패키지 설치
RUN apt-get update -y && \
    apt-get install -y python3 python3-pip git git-lfs curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Python 설정
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    python3 -m pip install --upgrade pip

# git-lfs 설정
RUN git lfs install

FROM base as builder

# 작업 디렉토리 설정
WORKDIR /workspace/anisoraV2_gpu

# 요구사항 파일 복사 및 설치
COPY ./anisoraV2_gpu/req-fastvideo.txt ./
COPY ./anisoraV2_gpu/requirements.txt ./
RUN python3 -m pip install --no-cache-dir -r req-fastvideo.txt && \
    python3 -m pip install --no-cache-dir -r requirements.txt && \
    python3 -m pip install --no-cache-dir streamlit

FROM base as final

# 빌더에서 설치된 패키지 복사
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 작업 디렉토리 설정
WORKDIR /workspace/anisoraV2_gpu

RUN git clone https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P

# 환경변수 설정
ENV NVIDIA_VISIBLE_DEVICES=all
ENV PYTHONPATH=/workspace

# 포트 노출
EXPOSE 8501

# 기본 명령어
CMD ["/bin/bash"]
