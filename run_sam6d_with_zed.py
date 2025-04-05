import os
import subprocess
import argparse
import time
import json
from pathlib import Path
import shutil

def run_command(cmd, cwd=None):
    """명령어를 실행하고 출력을 반환합니다."""
    print(f"실행 중: {cmd}")
    try:
        result = subprocess.run(cmd, cwd=cwd, shell=True, check=True, 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                text=True)
        print(result.stdout)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"오류 발생: {e}")
        print(f"표준 출력: {e.stdout}")
        print(f"표준 에러: {e.stderr}")
        return False, e.stderr

def main(args):
    base_dir = os.path.abspath(os.path.dirname(__file__))
    sam6d_dir = os.path.join(base_dir, "SAM-6D")
    
    # 1. 출력 디렉토리 설정
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        output_dir = os.path.join(base_dir, f"zed_sam6d_output_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"출력 디렉토리: {output_dir}")
    
    # 2. CAD 모델 확인
    if not args.cad_path:
        print("오류: CAD 모델 경로를 지정해야 합니다 (--cad_path)")
        return
    
    cad_path = os.path.abspath(args.cad_path)
    if not os.path.exists(cad_path) or not cad_path.endswith(".ply"):
        print(f"오류: CAD 모델 파일을 찾을 수 없거나 .ply 형식이 아닙니다: {cad_path}")
        return
    print(f"CAD 모델 경로: {cad_path}")
    
    # 3. Conda 환경 확인
    if args.conda_env:
        conda_env = args.conda_env
    else:
        conda_env = "sam6d"
    print(f"사용할 Conda 환경: {conda_env}")
    
    # 4. 단계별 처리 모드 설정
    capture_only = args.capture_only
    skip_capture = args.skip_capture
    if capture_only and skip_capture:
        print("오류: --capture_only와 --skip_capture는 동시에 사용할 수 없습니다.")
        return
    
    # 5. 작업 디렉토리 설정
    working_dir = os.path.join(output_dir, "sam6d_data")
    os.makedirs(working_dir, exist_ok=True)
    
    # 데이터 경로 설정
    rgb_path = os.path.join(working_dir, "rgb.png")
    depth_path = os.path.join(working_dir, "depth.png")
    camera_path = os.path.join(working_dir, "camera.json")
    results_dir = os.path.join(working_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 6. ZED 카메라 캡처 (옵션에 따라 건너뛸 수 있음)
    if not skip_capture:
        print("\n=== ZED 카메라 캡처 시작 ===")
        
        # ZED 캡처 명령 구성
        capture_cmd = f"python zed_capture.py --output_dir {working_dir} --cad_path {cad_path}"
        if args.save_original:
            capture_cmd += " --save_original"
        
        # Conda 환경에서 ZED 캡처 실행
        activate_cmd = f"conda activate {conda_env} && {capture_cmd}"
        
        if os.name == 'nt':  # Windows
            cmd = f"powershell -Command \"& {{conda activate {conda_env}; if ($?) {{ {capture_cmd} }}}}\""
        else:  # Linux/macOS
            cmd = f"bash -c 'source $(conda info --base)/etc/profile.d/conda.sh && {activate_cmd}'"
        
        success, _ = run_command(cmd)
        if not success:
            print("ZED 카메라 캡처 실패")
            return
        
        # 캡처만 하는 모드라면 여기서 종료
        if capture_only:
            print("\n=== 캡처 완료 (캡처만 모드) ===")
            print(f"캡처된 데이터 위치: {working_dir}")
            return
    else:
        # 캡처를 건너뛰는 경우, 입력 파일 확인
        if not all(os.path.exists(p) for p in [args.rgb_path, args.depth_path, args.camera_path]):
            print("오류: --skip_capture 옵션을 사용할 경우 --rgb_path, --depth_path, --camera_path를 모두 지정해야 합니다.")
            return
        
        # 입력 파일 복사
        print("\n=== 기존 데이터 사용 모드 ===")
        for src, dst in [(args.rgb_path, rgb_path), 
                         (args.depth_path, depth_path), 
                         (args.camera_path, camera_path)]:
            shutil.copy2(src, dst)
            print(f"파일 복사: {src} -> {dst}")
    
    # 7. SAM-6D 실행
    print("\n=== SAM-6D 파이프라인 실행 ===")
    
    # 파일 존재 여부 다시 확인
    for path, name in [(rgb_path, "RGB 이미지"), 
                       (depth_path, "Depth 맵"), 
                       (camera_path, "카메라 파라미터")]:
        if not os.path.exists(path):
            print(f"오류: {name} 파일을 찾을 수 없습니다: {path}")
            return
    
    # 환경 변수 설정
    env_vars = {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "SEGMENTOR_MODEL": args.segmentor_model,
        "CAD_PATH": cad_path,
        "RGB_PATH": rgb_path,
        "DEPTH_PATH": depth_path,
        "CAMERA_PATH": camera_path,
        "OUTPUT_DIR": results_dir
    }
    
    env_vars_str = " ".join([f"{k}={v}" for k, v in env_vars.items()])
    
    # SAM-6D 각 단계 실행
    if os.name == 'nt':  # Windows
        activate_cmd = f"conda activate {conda_env}"
    else:  # Linux/macOS
        activate_cmd = f"source $(conda info --base)/etc/profile.d/conda.sh && conda activate {conda_env}"
    
    # 1) CAD 템플릿 렌더링
    print("\n--- 1단계: CAD 템플릿 렌더링 ---")
    render_cmd = f"cd {os.path.join(sam6d_dir, 'Render')} && blenderproc run render_custom_templates.py --output_dir {results_dir} --cad_path {cad_path}"
    
    if os.name == 'nt':  # Windows
        cmd = f"powershell -Command \"& {{conda activate {conda_env}; if ($?) {{ {render_cmd} }}}}\""
    else:  # Linux/macOS
        cmd = f"bash -c '{activate_cmd} && {render_cmd}'"
    
    success, _ = run_command(cmd)
    if not success:
        print("CAD 템플릿 렌더링 실패")
        return
    
    # 2) 인스턴스 분할 모델 실행
    print("\n--- 2단계: 인스턴스 분할 모델 (ISM) ---")
    ism_cmd = f"cd {os.path.join(sam6d_dir, 'Instance_Segmentation_Model')} && python run_inference_custom.py --segmentor_model {args.segmentor_model} --output_dir {results_dir} --cad_path {cad_path} --rgb_path {rgb_path} --depth_path {depth_path} --cam_path {camera_path}"
    
    if os.name == 'nt':  # Windows
        cmd = f"powershell -Command \"& {{conda activate {conda_env}; if ($?) {{ {ism_cmd} }}}}\""
    else:  # Linux/macOS
        cmd = f"bash -c '{activate_cmd} && {ism_cmd}'"
    
    success, _ = run_command(cmd)
    if not success:
        print("인스턴스 분할 모델 실행 실패")
        return
    
    # 3) 포즈 추정 모델 실행
    print("\n--- 3단계: 포즈 추정 모델 (PEM) ---")
    seg_path = os.path.join(results_dir, "sam6d_results", "detection_ism.json")
    
    if not os.path.exists(seg_path):
        print(f"오류: 분할 결과 파일을 찾을 수 없습니다: {seg_path}")
        return
    
    pem_cmd = f"cd {os.path.join(sam6d_dir, 'Pose_Estimation_Model')} && python run_inference_custom.py --output_dir {results_dir} --cad_path {cad_path} --rgb_path {rgb_path} --depth_path {depth_path} --cam_path {camera_path} --seg_path {seg_path}"
    
    if os.name == 'nt':  # Windows
        cmd = f"powershell -Command \"& {{conda activate {conda_env}; if ($?) {{ {pem_cmd} }}}}\""
    else:  # Linux/macOS
        cmd = f"bash -c '{activate_cmd} && {pem_cmd}'"
    
    success, _ = run_command(cmd)
    if not success:
        print("포즈 추정 모델 실행 실패")
        return
    
    # 8. 결과 정리
    result_files = []
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith(('.png', '.json', '.txt')) and "sam6d_results" in root:
                result_files.append(os.path.join(root, file))
    
    # 결과 요약 파일 생성
    summary_path = os.path.join(output_dir, "results_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"SAM-6D 처리 결과 요약\n")
        f.write(f"처리 시간: {timestamp}\n\n")
        f.write(f"CAD 모델: {cad_path}\n")
        f.write(f"RGB 이미지: {rgb_path}\n")
        f.write(f"Depth 맵: {depth_path}\n")
        f.write(f"카메라 파라미터: {camera_path}\n\n")
        f.write(f"결과 파일 목록:\n")
        for file in result_files:
            rel_path = os.path.relpath(file, output_dir)
            f.write(f"- {rel_path}\n")
    
    print("\n=== SAM-6D 처리 완료 ===")
    print(f"결과 저장 위치: {results_dir}")
    print(f"결과 요약 파일: {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZED 카메라를 사용한 SAM-6D 통합 파이프라인")
    
    # 필수 인자
    parser.add_argument("--cad_path", type=str, help="CAD 모델 경로 (.ply 파일)")
    
    # 옵션 인자
    parser.add_argument("--output_dir", type=str, help="결과 저장 디렉토리 (기본값: zed_sam6d_output_<타임스탬프>)")
    parser.add_argument("--conda_env", type=str, default="sam6d", help="사용할 Conda 환경 (기본값: sam6d)")
    parser.add_argument("--segmentor_model", type=str, default="sam", choices=["sam", "fastsam"], 
                        help="사용할 분할 모델 (기본값: sam)")
    parser.add_argument("--save_original", action="store_true", help="원본 해상도 이미지도 저장")
    
    # 처리 모드 옵션
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--capture_only", action="store_true", help="ZED 카메라 캡처만 수행 (SAM-6D 파이프라인 실행 안 함)")
    group.add_argument("--skip_capture", action="store_true", help="ZED 카메라 캡처 단계 건너뛰기 (기존 데이터 사용)")
    
    # skip_capture 사용 시 필요한 옵션
    parser.add_argument("--rgb_path", type=str, help="기존 RGB 이미지 경로 (--skip_capture와 함께 사용)")
    parser.add_argument("--depth_path", type=str, help="기존 Depth 맵 경로 (--skip_capture와 함께 사용)")
    parser.add_argument("--camera_path", type=str, help="기존 카메라 파라미터 경로 (--skip_capture와 함께 사용)")
    
    args = parser.parse_args()
    
    main(args) 