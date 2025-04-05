#!/usr/bin/env python
"""
ZED 카메라와 SAM-6D 설정 테스트 스크립트
이 스크립트는 ZED 카메라 연결과 SAM-6D 환경이 올바르게 구성되었는지 확인합니다.
"""

import os
import sys
import subprocess
import importlib
import numpy as np
import platform
import json
from pathlib import Path

def check_module_exists(module_name):
    """모듈이 설치되어 있는지 확인합니다."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

def run_command(cmd):
    """명령어를 실행하고 출력을 반환합니다."""
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def print_section(title):
    """섹션 제목을 출력합니다."""
    print("\n" + "=" * 50)
    print(f" {title}")
    print("=" * 50)

def print_status(name, status, details=""):
    """상태를 출력합니다."""
    status_str = "✅ 성공" if status else "❌ 실패"
    print(f"{status_str} | {name}")
    if details and not status:
        print(f"  ↳ 세부 정보: {details}")
    return status

def test_zed_camera():
    """ZED 카메라 연결 및 설정을 테스트합니다."""
    print_section("ZED 카메라 테스트")
    
    # ZED SDK 설치 확인
    pyzed_installed = check_module_exists("pyzed")
    print_status("ZED Python SDK 설치", pyzed_installed, 
                "ZED SDK Python API가 설치되어 있지 않습니다. ZED SDK 설치 후 'python get_python_api.py'를 실행하세요.")
    
    if not pyzed_installed:
        return False
    
    # ZED 카메라 연결 테스트
    try:
        import pyzed.sl as sl
        
        # ZED 카메라 초기화
        zed = sl.Camera()
        init_params = sl.InitParameters()
        
        # 카메라 열기 시도
        err = zed.open(init_params)
        camera_connected = (err == sl.ERROR_CODE.SUCCESS)
        
        if camera_connected:
            # 카메라 정보 출력
            camera_info = zed.get_camera_information()
            model_name = camera_info.camera_model
            serial_number = camera_info.serial_number
            firmware = camera_info.firmware_version
            resolution = camera_info.camera_configuration.resolution
            fps = camera_info.camera_configuration.fps
            
            print_status("ZED 카메라 연결", True)
            print(f"  ↳ 모델: {model_name}")
            print(f"  ↳ 시리얼 번호: {serial_number}")
            print(f"  ↳ 펌웨어 버전: {firmware}")
            print(f"  ↳ 해상도: {resolution.width}x{resolution.height} @ {fps}fps")
            
            # 캘리브레이션 파라미터 확인
            calibration_params = camera_info.camera_configuration.calibration_parameters.left_cam
            print(f"  ↳ 카메라 내부 파라미터 (fx, fy, cx, cy): {calibration_params.fx:.2f}, {calibration_params.fy:.2f}, {calibration_params.cx:.2f}, {calibration_params.cy:.2f}")
            
            # 카메라 닫기
            zed.close()
            return True
        else:
            err_message = repr(err)
            print_status("ZED 카메라 연결", False, f"카메라 연결 실패: {err_message}")
            return False
    except Exception as e:
        print_status("ZED 카메라 연결", False, f"예외 발생: {str(e)}")
        return False

def test_sam6d_environment():
    """SAM-6D 환경 설정을 테스트합니다."""
    print_section("SAM-6D 환경 테스트")
    
    # 현재 디렉토리 확인
    cwd = os.getcwd()
    print(f"현재 작업 디렉토리: {cwd}")
    
    # SAM-6D 디렉토리 확인
    sam6d_dir = os.path.join(cwd, "SAM-6D")
    sam6d_exists = os.path.isdir(sam6d_dir)
    print_status("SAM-6D 디렉토리 확인", sam6d_exists, 
                f"SAM-6D 디렉토리를 찾을 수 없습니다: {sam6d_dir}")
    
    if not sam6d_exists:
        return False
    
    # 필수 파일 확인
    required_files = [
        "environment.yaml",
        "demo.sh",
        "prepare.sh"
    ]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(sam6d_dir, f))]
    files_exist = len(missing_files) == 0
    print_status("SAM-6D 필수 파일 확인", files_exist, 
                f"누락된 파일: {', '.join(missing_files)}" if missing_files else "")
    
    # 필수 디렉토리 확인
    required_dirs = [
        "Render",
        "Instance_Segmentation_Model",
        "Pose_Estimation_Model",
        "Data"
    ]
    missing_dirs = [d for d in required_dirs if not os.path.isdir(os.path.join(sam6d_dir, d))]
    dirs_exist = len(missing_dirs) == 0
    print_status("SAM-6D 필수 디렉토리 확인", dirs_exist, 
                f"누락된 디렉토리: {', '.join(missing_dirs)}" if missing_dirs else "")
    
    # Conda 환경 확인
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    is_sam6d_env = conda_env == "sam6d"
    print_status("SAM-6D Conda 환경 확인", is_sam6d_env, 
                f"현재 Conda 환경은 '{conda_env}'입니다. 'conda activate sam6d'를 실행하세요.")
    
    # 필수 Python 패키지 확인
    required_packages = [
        "torch", "torchvision", "opencv-cv2", "numpy", "xformers", 
        "pytorch_lightning", "distinctipy", "blenderproc", "trimesh"
    ]
    
    missing_packages = []
    for package in required_packages:
        if not check_module_exists(package.replace("-", "_")):
            missing_packages.append(package)
    
    packages_exist = len(missing_packages) == 0
    print_status("필수 Python 패키지 확인", packages_exist, 
                f"누락된 패키지: {', '.join(missing_packages)}" if missing_packages else "")
    
    # CUDA 확인
    if check_module_exists("torch"):
        import torch
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if cuda_available else "없음"
        device_count = torch.cuda.device_count() if cuda_available else 0
        device_name = torch.cuda.get_device_name(0) if cuda_available and device_count > 0 else "없음"
        
        print_status("CUDA 사용 가능", cuda_available, "CUDA를 사용할 수 없습니다.")
        if cuda_available:
            print(f"  ↳ CUDA 버전: {cuda_version}")
            print(f"  ↳ GPU 개수: {device_count}")
            print(f"  ↳ GPU 모델: {device_name}")
    else:
        print_status("CUDA 확인", False, "torch 패키지가 설치되어 있지 않아 CUDA를 확인할 수 없습니다.")
    
    # SAM-6D 커스텀 스크립트 확인
    custom_scripts = [
        "zed_capture.py",
        "run_sam6d_with_zed.py",
        "test_zed_sam6d_setup.py"
    ]
    
    existing_scripts = [s for s in custom_scripts if os.path.exists(os.path.join(cwd, s))]
    print(f"ZED-SAM-6D 연동 스크립트 {len(existing_scripts)}/{len(custom_scripts)} 개 발견")
    for script in existing_scripts:
        print(f"  ↳ {script}")
    
    return sam6d_exists and files_exist and dirs_exist

def test_example_data():
    """SAM-6D 예제 데이터가 있는지 확인합니다."""
    print_section("SAM-6D 예제 데이터 테스트")
    
    example_dir = os.path.join(os.getcwd(), "SAM-6D", "Data", "Example")
    example_exists = os.path.isdir(example_dir)
    print_status("예제 데이터 디렉토리 확인", example_exists, 
                f"예제 데이터 디렉토리를 찾을 수 없습니다: {example_dir}")
    
    if not example_exists:
        return False
    
    # 예제 파일 확인
    required_files = [
        "rgb.png",
        "depth.png",
        "camera.json"
    ]
    
    # CAD 모델 확인 (여러 파일 형식 가능)
    cad_exists = False
    cad_file = None
    for file in os.listdir(example_dir):
        if file.endswith(".ply") and os.path.isfile(os.path.join(example_dir, file)):
            cad_exists = True
            cad_file = file
            break
    
    print_status("CAD 모델 확인", cad_exists, 
                f"CAD 모델(.ply)을 찾을 수 없습니다." if not cad_exists else "")
    
    if cad_exists:
        print(f"  ↳ CAD 모델: {cad_file}")
    
    # 기타 필수 파일 확인
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(example_dir, f))]
    files_exist = len(missing_files) == 0
    print_status("기타 필수 예제 파일 확인", files_exist, 
                f"누락된 파일: {', '.join(missing_files)}" if missing_files else "")
    
    # 카메라 파라미터 파일 내용 확인
    if "camera.json" not in missing_files:
        camera_path = os.path.join(example_dir, "camera.json")
        try:
            with open(camera_path, 'r') as f:
                camera_data = json.load(f)
            
            has_cam_k = "cam_K" in camera_data
            print_status("카메라 파라미터 형식 확인", has_cam_k, 
                        f"카메라 파라미터 파일에 'cam_K' 항목이 없습니다.")
            
            if has_cam_k:
                cam_k = camera_data["cam_K"]
                if len(cam_k) == 9:
                    print(f"  ↳ 카메라 행렬 cam_K: [{cam_k[0]:.1f}, {cam_k[1]:.1f}, {cam_k[2]:.1f}, {cam_k[3]:.1f}, {cam_k[4]:.1f}, {cam_k[5]:.1f}, {cam_k[6]:.1f}, {cam_k[7]:.1f}, {cam_k[8]:.1f}]")
        except Exception as e:
            print_status("카메라 파라미터 형식 확인", False, f"카메라 파라미터 파일을 읽는 중 오류 발생: {str(e)}")
    
    return example_exists and cad_exists and files_exist

def main():
    """메인 함수."""
    print_section("시스템 정보")
    print(f"Python 버전: {platform.python_version()}")
    print(f"운영체제: {platform.system()} {platform.version()}")
    print(f"머신: {platform.machine()}")
    
    # conda 환경 확인
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        print(f"Conda 환경: {os.path.basename(conda_prefix)}")
    else:
        print("Conda 환경: 감지되지 않음")
    
    # 각 테스트 실행
    zed_test_result = test_zed_camera()
    sam6d_test_result = test_sam6d_environment()
    example_test_result = test_example_data()
    
    # 종합 결과
    print_section("테스트 종합 결과")
    
    all_passed = all([zed_test_result, sam6d_test_result, example_test_result])
    if all_passed:
        print("✅ 모든 테스트 통과! ZED 카메라와 SAM-6D 환경이 올바르게 설정되었습니다.")
        print("\n다음 명령으로 ZED 카메라 데이터를 캡처하고 SAM-6D를 실행할 수 있습니다:")
        
        example_dir = os.path.join(os.getcwd(), "SAM-6D", "Data", "Example")
        cad_path = None
        for file in os.listdir(example_dir):
            if file.endswith(".ply"):
                cad_path = os.path.join(example_dir, file)
                break
        
        if cad_path:
            print(f"\npython run_sam6d_with_zed.py --cad_path {cad_path}")
        else:
            print("\npython run_sam6d_with_zed.py --cad_path <your_cad_model.ply>")
    else:
        failed_tests = []
        if not zed_test_result:
            failed_tests.append("ZED 카메라 테스트")
        if not sam6d_test_result:
            failed_tests.append("SAM-6D 환경 테스트")
        if not example_test_result:
            failed_tests.append("예제 데이터 테스트")
        
        print(f"❌ 일부 테스트 실패: {', '.join(failed_tests)}")
        print("\n문제 해결 방법은 ZED_SAM6D_README.md 파일을 참조하세요.")

if __name__ == "__main__":
    main() 