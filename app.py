# app.py
import streamlit as st
import subprocess
import os
import sys
from datetime import datetime

# 페이지 설정
st.set_page_config(
    page_title="AI Video Generation Interface",
    page_icon="🎬",
    layout="wide"
)

# 제목
st.title("🎬 AI Video Generation Interface")
st.markdown("---")

# 사이드바에서 환경변수 설정
st.sidebar.header("🔧 환경변수 설정")

# 기본 환경변수들
default_env_vars = {
    "CUDA_VISIBLE_DEVICES": "0",
    "PYTHONPATH": "/workspace",
    "MODEL_PATH": "/workspace/anisoraV2_gpu/Wan2.1-I2V-14B-480P",
    "OUTPUT_DIR": "/workspace/anisoraV2_gpu/output_videos",
    "TORCH_HOME": "/workspace/.cache/torch"
}

# 환경변수 입력 섹션
env_vars = {}
st.sidebar.subheader("기본 환경변수")
for key, default_value in default_env_vars.items():
    env_vars[key] = st.sidebar.text_input(
        key, 
        value=default_value,
        help=f"환경변수 {key}의 값을 설정하세요"
    )

# 추가 환경변수
st.sidebar.subheader("추가 환경변수")
additional_env = st.sidebar.text_area(
    "추가 환경변수 (KEY=VALUE 형식, 한 줄에 하나씩)",
    height=100,
    placeholder="CUSTOM_VAR=value1\nANOTHER_VAR=value2"
)

# 메인 화면 - 명령어 파라미터 설정
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📝 명령어 파라미터 설정")
    
    # 기본 파라미터들
    task = st.selectbox(
        "Task",
        options=["i2v-14B", "t2v-14B", "i2v-7B"],
        index=0,
        help="실행할 작업 유형을 선택하세요"
    )
    
    size = st.selectbox(
        "Size",
        options=["960*544", "1280*720", "640*480", "512*512"],
        index=0,
        help="생성할 비디오의 해상도를 선택하세요"
    )
    
    ckpt_dir = st.text_input(
        "Checkpoint Directory",
        value="Wan2.1-I2V-14B-480P",
        help="모델 체크포인트 디렉토리 경로"
    )
    
    image_path = st.text_input(
        "Image Path",
        value="output_videos",
        help="입력 이미지 경로 또는 출력 디렉토리"
    )
    
    base_seed = st.number_input(
        "Base Seed",
        min_value=0,
        max_value=999999,
        value=4096,
        help="랜덤 시드 값"
    )
    
    frame_num = st.number_input(
        "Frame Number",
        min_value=1,
        max_value=200,
        value=49,
        help="생성할 프레임 수"
    )

with col2:
    st.header("🎯 프롬프트 설정")
    
    # 프롬프트 입력 방식 선택
    prompt_mode = st.radio(
        "프롬프트 입력 방식",
        ["직접 입력", "파일 경로"]
    )
    
    if prompt_mode == "직접 입력":
        prompt_text = st.text_area(
            "프롬프트 텍스트",
            value="A beautiful sunset over the ocean with gentle waves",
            height=150,
            help="생성하고 싶은 비디오에 대한 설명을 입력하세요"
        )
        prompt_file = None
    else:
        prompt_file = st.text_input(
            "프롬프트 파일 경로",
            value="data/inference.txt",
            help="프롬프트가 저장된 텍스트 파일 경로"
        )
        prompt_text = None

# 고급 옵션
with st.expander("🔧 고급 옵션"):
    col3, col4 = st.columns(2)
    
    with col3:
        additional_args = st.text_input(
            "추가 인수",
            placeholder="--cfg_scale 7.5 --steps 50",
            help="추가로 전달할 명령행 인수들"
        )
    
    with col4:
        working_dir = st.text_input(
            "작업 디렉토리",
            value="/workspace/anisoraV2_gpu",
            help="명령어를 실행할 작업 디렉토리"
        )

# 명령어 미리보기
st.header("👀 생성될 명령어 미리보기")

# 명령어 구성
command_parts = [
    "python", "generate-pi-i2v.py",
    f"--task {task}",
    f"--size {size}",
    f"--ckpt_dir {ckpt_dir}",
    f"--image {image_path}",
    f"--base_seed {base_seed}",
    f"--frame_num {frame_num}"
]

if prompt_mode == "직접 입력" and prompt_text:
    # 임시 파일에 프롬프트 저장
    temp_prompt_file = "/tmp/temp_prompt.txt"
    command_parts.append(f"--prompt {temp_prompt_file}")
elif prompt_mode == "파일 경로" and prompt_file:
    command_parts.append(f"--prompt {prompt_file}")

if additional_args:
    command_parts.append(additional_args)

full_command = " ".join(command_parts)

st.code(full_command, language="bash")

# 실행 버튼과 결과 표시
col5, col6 = st.columns([1, 4])

with col5:
    if st.button("🚀 실행", type="primary", use_container_width=True):
        st.session_state.execute_command = True
        st.session_state.command = full_command
        st.session_state.prompt_text = prompt_text
        st.session_state.env_vars = env_vars
        st.session_state.additional_env = additional_env
        st.session_state.working_dir = working_dir

with col6:
    if st.button("🧹 초기화", use_container_width=True):
        st.rerun()

# 명령어 실행 및 결과 표시
if hasattr(st.session_state, 'execute_command') and st.session_state.execute_command:
    st.header("📊 실행 결과")
    
    # 환경변수 준비
    env_dict = os.environ.copy()
    
    # 기본 환경변수 추가
    for key, value in st.session_state.env_vars.items():
        if value.strip():
            env_dict[key] = value.strip()
    
    # 추가 환경변수 파싱
    if st.session_state.additional_env.strip():
        for line in st.session_state.additional_env.strip().split('\n'):
            if '=' in line and line.strip():
                key, value = line.split('=', 1)
                env_dict[key.strip()] = value.strip()
    
    # 프롬프트 텍스트를 임시 파일에 저장 (직접 입력인 경우)
    if st.session_state.prompt_text:
        try:
            with open("/tmp/temp_prompt.txt", "w", encoding="utf-8") as f:
                f.write(st.session_state.prompt_text)
        except Exception as e:
            st.error(f"프롬프트 파일 생성 오류: {str(e)}")
    
    # 실행 상태 표시
    with st.spinner("명령어 실행 중... 시간이 오래 걸릴 수 있습니다."):
        try:
            # 명령어 실행
            process = subprocess.Popen(
                st.session_state.command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env_dict,
                cwd=st.session_state.working_dir
            )
            
            # 실시간 출력 표시
            stdout_container = st.empty()
            stderr_container = st.empty()
            
            stdout_lines = []
            stderr_lines = []
            
            # 프로세스 실행 중 실시간 출력 업데이트
            while True:
                stdout_line = process.stdout.readline()
                stderr_line = process.stderr.readline()
                
                if stdout_line:
                    stdout_lines.append(stdout_line.strip())
                    stdout_container.text_area(
                        "표준 출력 (실시간)", 
                        "\n".join(stdout_lines[-50:]),  # 최근 50줄만 표시
                        height=200
                    )
                
                if stderr_line:
                    stderr_lines.append(stderr_line.strip())
                    stderr_container.text_area(
                        "오류 출력 (실시간)", 
                        "\n".join(stderr_lines[-50:]),  # 최근 50줄만 표시
                        height=200
                    )
                
                if process.poll() is not None:
                    break
            
            # 프로세스 완료 후 최종 결과
            return_code = process.returncode
            
            # 결과 표시
            if return_code == 0:
                st.success(f"✅ 명령어가 성공적으로 실행되었습니다! (종료 코드: {return_code})")
            else:
                st.error(f"❌ 명령어 실행 중 오류가 발생했습니다. (종료 코드: {return_code})")
            
            # 실행 시간 기록
            st.info(f"🕐 실행 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            st.error(f"실행 중 예외 발생: {str(e)}")
        
        finally:
            # 실행 상태 초기화
            st.session_state.execute_command = False

# 사용법 안내
with st.expander("ℹ️ 사용법 안내"):
    st.markdown("""
    ### 사용 방법
    1. **환경변수 설정**: 좌측 사이드바에서 필요한 환경변수를 설정하세요
    2. **파라미터 설정**: 메인 화면에서 명령어 파라미터들을 설정하세요
    3. **프롬프트 입력**: 직접 텍스트를 입력하거나 파일 경로를 지정하세요
    4. **명령어 확인**: 생성된 명령어를 미리보기에서 확인하세요
    5. **실행**: '실행' 버튼을 클릭하여 명령어를 실행하세요
    
    ### 주요 기능
    - 🎯 직관적인 파라미터 설정
    - 📝 프롬프트 직접 입력 또는 파일 지정
    - 🔧 환경변수 커스터마이징
    - 👀 실시간 실행 결과 모니터링
    - 🚀 원클릭 실행
    
    ### 팁
    - 프롬프트는 영어로 입력하는 것이 좋습니다
    - 긴 실행 시간이 예상되므로 인내심을 가지고 기다려주세요
    - 오류 발생 시 오류 출력을 확인하여 문제를 파악하세요
    """)
