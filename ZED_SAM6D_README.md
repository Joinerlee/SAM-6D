# ZED 카메라와 SAM-6D 연동 가이드

이 문서는 ZED 카메라에서 캡처한 데이터를 SAM-6D 파이프라인에 연동하는 방법을 설명합니다.

## 📋 목차

1. [소개](#소개)
2. [설치 및 설정](#설치-및-설정)
3. [사용 방법](#사용-방법)
4. [문제 해결](#문제-해결)
5. [유의사항](#유의사항)

## 📝 소개

SAM-6D는 단일 RGB-D 이미지에서 객체의 6D 포즈를 추정하는 프레임워크입니다. 이 프로젝트는 ZED 스테레오 카메라의 RGB-D 데이터를 SAM-6D에 연동하여 실시간으로 객체의 6D 포즈를 추정할 수 있도록 합니다.

주요 기능:
- ZED 카메라에서 RGB 이미지와 Depth 맵 캡처
- SAM-6D에 적합한 형식(640x480)으로 변환
- CAD 모델을 사용한 객체의 6D 포즈 추정

## 🔧 설치 및 설정

### 필수 요구사항

- NVIDIA GPU (CUDA 지원)
- Ubuntu 22.04 (권장) 또는 Windows 10/11
- Anaconda 설치
- ZED 카메라
- ZED SDK 설치
- SAM-6D 저장소 복제

### 환경 설정

1. **SAM-6D 복제 및 환경 생성**

```bash
# 저장소 복제
git clone https://github.com/savidini/SAM-6D.git

# 기존 environment.yaml 파일 수정 (CUDA 12.4 호환성 및 pyzed 추가)
# 수정된 environment.yaml 파일이 이미 준비되어 있습니다

# Conda 환경 생성
cd SAM-6D
conda env create -f environment.yaml
conda activate sam6d
```

2. **ZED SDK Python API 설치**

```bash
# Conda 환경 활성화
conda activate sam6d

# ZED SDK 디렉토리로 이동 (기본 경로, 다를 수 있음)
cd /usr/local/zed
python3 get_python_api.py
```

3. **SAM-6D 의존성 설치**

```bash
cd SAM-6D
sh prepare.sh
```

## 🚀 사용 방법

### 1. ZED 카메라 데이터 캡처

ZED 카메라에서 데이터를 캡처하고 SAM-6D에 적합한 형식으로 변환할 수 있습니다.

```bash
# Conda 환경 활성화
conda activate sam6d

# 데이터 캡처 (기본 출력 디렉토리: zed_output_<타임스탬프>)
python zed_capture.py

# 특정 출력 디렉토리 지정
python zed_capture.py --output_dir my_data

# 원본 이미지도 저장
python zed_capture.py --output_dir my_data --save_original

# CAD 모델 경로 지정 (알림용, 파일은 복사되지 않음)
python zed_capture.py --output_dir my_data --cad_path my_model.ply
```

캡처 프로그램 사용 방법:
- 's' 키: 현재 프레임 저장
- 'q' 키 또는 ESC: 프로그램 종료

### 2. 통합 파이프라인 실행

ZED 카메라 캡처부터 SAM-6D 파이프라인 실행까지 한 번에 수행합니다.

```bash
# 기본 실행 (CAD 모델 경로 필수)
python run_sam6d_with_zed.py --cad_path /path/to/model.ply

# 출력 디렉토리 지정
python run_sam6d_with_zed.py --cad_path /path/to/model.ply --output_dir my_results

# 분할 모델 선택 (sam 또는 fastsam)
python run_sam6d_with_zed.py --cad_path /path/to/model.ply --segmentor_model fastsam

# 캡처만 수행 (SAM-6D 실행 안 함)
python run_sam6d_with_zed.py --cad_path /path/to/model.ply --capture_only

# 캡처 단계 건너뛰기 (기존 데이터 사용)
python run_sam6d_with_zed.py --cad_path /path/to/model.ply --skip_capture \
    --rgb_path /path/to/rgb.png \
    --depth_path /path/to/depth.png \
    --camera_path /path/to/camera.json
```

## ⚠️ 문제 해결

### CUDA 버전 문제

기본 SAM-6D 환경은 CUDA 12.1을 지정하고 있으나, 시스템에 CUDA 12.4가 설치되어 있을 수 있습니다. 이 경우 다음과 같이 해결하세요:

1. `environment.yaml` 파일에서 모든 CUDA 12.1 참조를 CUDA 12.4로 변경
2. 환경을 새로 생성하거나 기존 환경 업데이트:
   ```bash
   # 새로운 환경 생성
   conda env create -f environment.yaml
   
   # 또는 기존 환경 업데이트
   conda activate sam6d
   conda env update -f environment.yaml
   ```

### ZED 카메라 인식 문제

ZED 카메라가 인식되지 않는 경우:

1. USB 연결 상태 확인
2. ZED SDK가 올바르게 설치되었는지 확인
3. ZED Python API가 현재 conda 환경에 설치되었는지 확인:
   ```bash
   conda activate sam6d
   pip list | grep pyzed
   ```

### SAM-6D 실행 오류

SAM-6D 파이프라인 실행 중 오류가 발생하면:

1. 환경 변수가 올바르게 설정되었는지 확인
2. 입력 파일(RGB, Depth, 카메라 파라미터, CAD 모델)이 올바른 형식과 경로에 있는지 확인
3. SAM-6D 의존성이 모두 설치되었는지 확인 (prepare.sh 스크립트 실행)

## 📋 유의사항

1. **ZED 카메라 해상도**: ZED 카메라는 기본적으로 HD720(1280x720) 해상도로 설정되어 있으며, 이 데이터는 SAM-6D에 맞게 640x480으로 리사이징됩니다. 이 과정에서 카메라 내부 파라미터도 적절히 조정됩니다.

2. **Depth 단위**: ZED 카메라와 SAM-6D 모두 밀리미터(mm) 단위를 사용하도록 설정되어 있습니다.

3. **CAD 모델**: SAM-6D는 .ply 형식의 CAD 모델을 필요로 합니다. 모델의 단위는 밀리미터(mm)여야 합니다.

4. **메모리 사용량**: SAM-6D 파이프라인은 상당한 양의 GPU 메모리를 사용합니다. VRAM이 충분한 GPU(최소 8GB, 권장 16GB 이상)를 사용하세요.

5. **실행 시간**: 전체 파이프라인은 GPU 성능에 따라 수 초에서 수십 초가 소요될 수 있습니다. 현재 구현은 실시간에 가깝지만 완전한 실시간은 아닙니다. 