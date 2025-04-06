import pyzed.sl as sl
import numpy as np
import cv2
import json
import os
import argparse
from datetime import datetime
import time

def main(args):
    # ZED 카메라 객체 생성
    zed = sl.Camera()

    # 초기화 파라미터 설정
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Depth 모드 설정
    init_params.coordinate_units = sl.UNIT.MILLIMETER  # SAM-6D를 위해 MILLIMETER 사용
    
    # 카메라 설정 추가
    init_params.camera_image_flip = sl.FLIP_MODE.OFF  # 이미지 플립 없음
    init_params.enable_right_side_measure = False  # 오른쪽 카메라 측정 비활성화 (필요없음)
    
    # 카메라를 기본 해상도로 설정 (VGA로 강제하지 않음)
    # init_params.camera_resolution = sl.RESOLUTION.VGA  # 주석 처리
    init_params.camera_fps = 30  # 프레임 속도 설정

    # 카메라 열기
    print("카메라 연결 중...")
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"ZED 카메라 열기 실패: {repr(err)}. 종료합니다...")
        zed.close()
        exit()

    # 카메라가 초기화될 시간 주기
    print("카메라 초기화 중... 잠시 기다려주세요")
    time.sleep(2)

    # 이미지와 Depth 데이터를 담을 sl.Mat 객체 생성
    image_zed = sl.Mat()
    depth_zed = sl.Mat()
    
    # 런타임 파라미터 설정
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.enable_depth = True  # 깊이 측정 활성화

    # 카메라 캘리브레이션 파라미터 가져오기 (왼쪽 카메라 기준)
    calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
    cam_k = [
        calibration_params.fx, 0.0, calibration_params.cx,
        0.0, calibration_params.fy, calibration_params.cy,
        0.0, 0.0, 1.0
    ]
    intrinsics_dict = {"cam_K": cam_k}
    print(f"카메라 내부 파라미터 (fx, fy, cx, cy): {calibration_params.fx}, {calibration_params.fy}, {calibration_params.cx}, {calibration_params.cy}")
    
    # 해상도 정보 확인
    img_width = zed.get_camera_information().camera_configuration.resolution.width
    img_height = zed.get_camera_information().camera_configuration.resolution.height
    print(f"카메라 기본 해상도: {img_width}x{img_height}")
    
    # SAM-6D 필요 해상도 확인
    print("중앙 기준으로 640x480 영역을 추출합니다.")
    
    # 크롭할 시작 위치 계산 (중앙 기준)
    start_x = max(0, (img_width - 640) // 2)
    start_y = max(0, (img_height - 480) // 2)
    
    # 내부 파라미터 조정 (크롭에 맞게)
    new_cx = calibration_params.cx - start_x  # cx 조정
    new_cy = calibration_params.cy - start_y  # cy 조정
    
    # 새로운 내부 파라미터 설정
    intrinsics_dict = {
        "cam_K": [
            calibration_params.fx, 0.0, new_cx,
            0.0, calibration_params.fy, new_cy,
            0.0, 0.0, 1.0
        ]
    }
    
    print(f"크롭 영역: ({start_x},{start_y})부터 640x480 크기")
    print(f"조정된 카메라 내부 파라미터: cx={new_cx}, cy={new_cy}")
    
    # CAD 모델 경로 확인 및 표시
    cad_path = args.cad_path
    if cad_path and os.path.exists(cad_path):
        print(f"CAD 모델: {cad_path}")
    elif cad_path:
        print(f"주의: 지정한 CAD 모델 파일({cad_path})을 찾을 수 없습니다.")
    else:
        print("CAD 모델 경로가 지정되지 않았습니다.")
    
    print("\n's' 키를 눌러 데이터를 저장하고, 'q' 또는 ESC 키를 눌러 종료하세요.")
    
    # 크롭할 시작 위치 계산 (중앙 기준) - 루프 밖에서 미리 계산
    end_x = min(img_width, start_x + 640)
    end_y = min(img_height, start_y + 480)
    crop_width = end_x - start_x
    crop_height = end_y - start_y
    
    # 이미지 크기가 충분히 큰지 확인
    if crop_width < 640 or crop_height < 480:
        print(f"주의: 카메라 해상도({img_width}x{img_height})가 SAM-6D 목표 해상도(640x480)보다 작습니다.")
        print(f"실제 크롭 크기: {crop_width}x{crop_height}")
    
    try:
        while True:
            # 새로운 프레임 잡기
            if zed.grab(runtime_parameters) != sl.ERROR_CODE.SUCCESS:
                print("프레임 획득 실패, 다시 시도합니다...")
                time.sleep(0.1)
                continue
                
            try:
                # 1. 이미지 데이터 가져오기 (기본 해상도로)
                retrieve_status = zed.retrieve_image(image_zed, sl.VIEW.LEFT)
                if retrieve_status != sl.ERROR_CODE.SUCCESS:
                    print(f"이미지 검색 실패: {retrieve_status}, 다시 시도합니다...")
                    continue
                
                # NumPy 배열로 이미지 데이터 복사
                image_rgb = image_zed.get_data()
                # RGBA 이미지를 BGR로 변환 (채널 수 확인)
                if image_rgb.shape[2] == 4:  # 4채널(RGBA) 이미지인 경우
                    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2BGR)
                
                # 2. 깊이 맵 가져오기 (기본 해상도로)
                retrieve_status = zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)
                if retrieve_status != sl.ERROR_CODE.SUCCESS:
                    print(f"깊이 맵 검색 실패: {retrieve_status}, 다시 시도합니다...")
                    continue
                
                # NumPy 배열로 깊이 데이터 복사
                depth_map = depth_zed.get_data()
                # 깊이 맵이 유효한지 확인하고 float32로 변환
                if depth_map is not None and depth_map.size > 0:
                    depth_map = depth_map.astype(np.float32)
                    # NaN 값과 무한대 값 처리
                    depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    print("유효한 깊이 데이터를 가져오지 못했습니다")
                    continue
                
                # 이미지 형태 확인 (디버깅용)
                if "first_frame" not in locals():
                    print(f"원본 이미지 형태: {image_rgb.shape}, 타입: {image_rgb.dtype}")
                    print(f"원본 깊이 맵 형태: {depth_map.shape}, 타입: {depth_map.dtype}")
                    first_frame = False
                
                # 중앙 640x480 영역 크롭
                if crop_width >= 640 and crop_height >= 480:
                    # 이미지 크기가 충분히 큰 경우 - 중앙 크롭
                    # 이미지가 3채널인지 확인
                    if len(image_rgb.shape) == 3 and image_rgb.shape[2] == 3:
                        sam6d_color = image_rgb[start_y:start_y+480, start_x:start_x+640].copy()
                        sam6d_depth = depth_map[start_y:start_y+480, start_x:start_x+640].copy()
                    else:
                        print("이미지 형식 오류: 3채널(BGR) 이미지가 아닙니다")
                        continue
                else:
                    # 이미지 크기가 작은 경우 - 패딩 처리
                    sam6d_color = np.zeros((480, 640, 3), dtype=np.uint8)
                    sam6d_depth = np.zeros((480, 640), dtype=np.float32)
                    
                    # 패딩된 이미지의 중앙에 원본 이미지 배치
                    pad_start_x = (640 - crop_width) // 2
                    pad_start_y = (480 - crop_height) // 2
                    
                    sam6d_color[pad_start_y:pad_start_y+crop_height, pad_start_x:pad_start_x+crop_width] = image_rgb[start_y:end_y, start_x:end_x].copy()
                    sam6d_depth[pad_start_y:pad_start_y+crop_height, pad_start_x:pad_start_x+crop_width] = depth_map[start_y:end_y, start_x:end_x].copy()
                
                # 표시용 Depth 이미지 생성
                try:
                    depth_display = np.zeros_like(sam6d_color)
                    if np.max(sam6d_depth) > np.min(sam6d_depth):
                        # 범위가 있는 데이터만 표시
                        depth_norm = cv2.normalize(sam6d_depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                        depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
                        depth_display = depth_colormap
                except Exception as e:
                    print(f"깊이 맵 시각화 오류: {str(e)}")
                
                # 이미지 및 깊이 맵 표시 준비
                try:
                    # 원본 이미지와 Depth 맵을 가로로 합치기
                    display_image = np.hstack((sam6d_color, depth_display))
                    
                    # 설명 텍스트 추가
                    cv2.putText(display_image, f"ZED: {img_width}x{img_height} → SAM-6D: 640x480 (중앙 크롭)", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
                    # 이미지 보여주기
                    cv2.imshow("ZED 카메라 | SAM-6D용 캡처", display_image)
                except Exception as e:
                    print(f"디스플레이 오류: {str(e)}")
                    continue

                # --- 키 처리 ---
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 종료
                    print("종료 중...")
                    break
                elif key == ord('s'):  # 데이터 저장
                    # 타임스탬프로 폴더명 생성
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    if args.output_dir:
                        output_dir = args.output_dir
                    else:
                        output_dir = f"zed_output_{timestamp}"
                    
                    os.makedirs(output_dir, exist_ok=True)

                    # 파일 경로 생성
                    rgb_path = os.path.join(output_dir, "rgb.png")
                    depth_path = os.path.join(output_dir, "depth.png")
                    camera_path = os.path.join(output_dir, "camera.json")
                    
                    # SAM-6D용 데이터 저장
                    cv2.imwrite(rgb_path, sam6d_color)
                    # 16비트 PNG로 깊이 맵 저장 (SAM-6D와 호환되도록)
                    cv2.imwrite(depth_path, sam6d_depth.astype(np.uint16))
                    with open(camera_path, "w") as f:
                        json.dump(intrinsics_dict, f, indent=4)
                    
                    print(f"SAM-6D용 데이터 저장 완료: {output_dir}")
                    
                    # CAD 모델 경로 출력
                    cad_path_info = args.cad_path if args.cad_path else "/home/recl3090/SAM-6D/TAN.ply (기본값)"
                    print(f"CAD 모델 경로: {cad_path_info}")
                    
                    # SAM-6D 실행 준비 완료 메시지
                    used_cad_path = args.cad_path if args.cad_path else "/home/recl3090/SAM-6D/TAN.ply"
                    print("\nSAM-6D 실행 환경변수 예시:")
                    print(f"export CAD_PATH={used_cad_path}")
                    print(f"export RGB_PATH={os.path.abspath(rgb_path)}")
                    print(f"export DEPTH_PATH={os.path.abspath(depth_path)}")
                    print(f"export CAMERA_PATH={os.path.abspath(camera_path)}")
                    print(f"export OUTPUT_DIR={os.path.abspath(output_dir)}/results")
                
            except Exception as e:
                print(f"처리 오류: {str(e)}")
                time.sleep(0.1)
                continue

    finally:
        # 카메라 닫고 창 제거
        zed.close()
        cv2.destroyAllWindows()
        print("ZED 카메라가 닫혔습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZED 카메라를 사용하여 SAM-6D용 데이터 캡처")
    parser.add_argument("--output_dir", type=str, help="출력 디렉토리 경로")
    parser.add_argument("--cad_path", type=str, default="/home/recl3090/SAM-6D/TAN.ply", help="CAD 모델 경로(.ply 파일)")
    args = parser.parse_args()
    
    main(args) 