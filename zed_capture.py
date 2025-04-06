import pyzed.sl as sl
import numpy as np
import cv2
import json
import os
import argparse
from datetime import datetime

def main(args):
    # ZED 카메라 객체 생성
    zed = sl.Camera()

    # 초기화 파라미터 설정
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Depth 모드 설정
    init_params.coordinate_units = sl.UNIT.MILLIMETER  # SAM-6D를 위해 MILLIMETER 사용
    
    # SAM-6D 대상 해상도로 직접 설정 (640x480)
    init_params.camera_resolution = sl.RESOLUTION.VGA  # VGA는 640x480 해상도
    init_params.camera_fps = 30  # 프레임 속도 설정

    # 카메라 열기
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"ZED 카메라 열기 실패: {repr(err)}. 종료합니다...")
        zed.close()
        exit()

    # 이미지와 Depth 데이터를 담을 sl.Mat 객체 생성
    image_zed = sl.Mat()
    depth_zed = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()

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
    print(f"카메라 해상도: {img_width}x{img_height}")
    
    # SAM-6D 필요 해상도 확인 (이미 VGA로 설정했으므로 640x480이어야 함)
    if img_width != 640 or img_height != 480:
        print(f"경고: 현재 해상도({img_width}x{img_height})가 SAM-6D 대상 해상도(640x480)와 다릅니다.")
        
        # SAM-6D 요구 해상도로 조정된 카메라 내부 파라미터 계산
        width_ratio = 640.0 / img_width
        height_ratio = 480.0 / img_height
        
        # 카메라 파라미터 조정
        target_cam_k = [
            cam_k[0] * width_ratio, cam_k[1], cam_k[2] * width_ratio,
            cam_k[3], cam_k[4] * height_ratio, cam_k[5] * height_ratio,
            cam_k[6], cam_k[7], cam_k[8]
        ]
        intrinsics_dict = {"cam_K": target_cam_k}
        print(f"SAM-6D 해상도(640x480)에 맞게 카메라 내부 파라미터를 조정했습니다.")
    else:
        print("SAM-6D 호환 해상도(640x480)로 설정되었습니다.")
    
    print("\n's' 키를 눌러 데이터를 저장하고, 'q' 또는 ESC 키를 눌러 종료하세요.")

    # 디스플레이 플래그 - 처음에만, 혹은 디버깅 정보가 필요할 때만 True로 설정
    show_debug_info = True
    
    try:
        while True:
            # 새로운 프레임 잡기
            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                # 이미지 및 Depth 데이터 검색 - 에러 처리와 함께
                try:
                    # 1. 이미지 검색
                    zed.retrieve_image(image_zed, sl.VIEW.LEFT)
                    
                    # 2. Depth 맵 검색
                    zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)
                    
                    # 3. NumPy 배열로 변환
                    color_image = image_zed.get_data()
                    depth_map = depth_zed.get_data()
                    
                    # 4. 데이터 유효성 검사
                    if color_image is None or depth_map is None or color_image.size == 0 or depth_map.size == 0:
                        print("유효하지 않은 카메라 데이터, 다시 시도합니다...")
                        continue
                    
                    # 4. RGBA에서 BGR로 변환 (필요한 경우)
                    if color_image.shape[2] == 4:  # RGBA 형식인 경우
                        color_image_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGBA2BGR)
                    else:  # 이미 BGR인 경우
                        color_image_bgr = color_image
                    
                    # 디버그 정보 (첫 번째 프레임이나 디버깅이 필요할 때만 표시)
                    if show_debug_info:
                        print(f"컬러 이미지 형태: {color_image_bgr.shape}, 타입: {color_image_bgr.dtype}")
                        print(f"깊이 맵 형태: {depth_map.shape}, 타입: {depth_map.dtype}")
                        show_debug_info = False  # 한 번만 표시
                        
                except Exception as e:
                    print(f"이미지 처리 오류: {str(e)}")
                    continue
                
                # SAM-6D 필요 해상도로 리사이징 (필요한 경우)
                if img_width != 640 or img_height != 480:
                    sam6d_color = cv2.resize(color_image_bgr, (640, 480))
                    sam6d_depth = cv2.resize(depth_map, (640, 480))
                else:
                    sam6d_color = color_image_bgr
                    sam6d_depth = depth_map
                
                # Depth 맵의 NaN/Inf 값 처리 (0으로 대체)
                sam6d_depth = np.nan_to_num(sam6d_depth, nan=0.0, posinf=0.0, neginf=0.0)
                
                # 표시용 Depth 이미지 생성
                try:
                    # 범위가 있는 데이터만 표시
                    if np.max(sam6d_depth) > np.min(sam6d_depth):
                        depth_display = cv2.normalize(sam6d_depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                        depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
                    else:
                        # 깊이 데이터가 모두 같은 경우 (일정한 패턴으로 초기화)
                        depth_colormap = np.zeros_like(sam6d_color)
                except Exception as e:
                    print(f"깊이 맵 시각화 오류: {str(e)}")
                    depth_colormap = np.zeros_like(sam6d_color)
                
                # 이미지 및 깊이 맵 표시 준비
                try:
                    # 이미지 크기 맞추기
                    if sam6d_color.shape[0:2] != depth_colormap.shape[0:2]:
                        depth_colormap = cv2.resize(depth_colormap, (sam6d_color.shape[1], sam6d_color.shape[0]))
                    
                    # 원본 이미지와 Depth 맵을 가로로 합치기
                    display_image = np.hstack((sam6d_color, depth_colormap))
                    
                    # 설명 텍스트 추가
                    cv2.putText(display_image, f"ZED: {img_width}x{img_height} | SAM-6D: 640x480", (10, 30), 
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
                    
                    # CAD 모델 확인 메시지
                    if args.cad_path:
                        print(f"CAD 모델 경로: {args.cad_path}")
                    else:
                        print("주의: CAD 모델 경로를 지정하지 않았습니다. SAM-6D 실행을 위해 CAD 모델이 필요합니다.")
                    
                    # SAM-6D 실행 준비 완료 메시지
                    print("\nSAM-6D 실행 환경변수 예시:")
                    print(f"export CAD_PATH=<your_cad_model_path.ply>")
                    print(f"export RGB_PATH={os.path.abspath(rgb_path)}")
                    print(f"export DEPTH_PATH={os.path.abspath(depth_path)}")
                    print(f"export CAMERA_PATH={os.path.abspath(camera_path)}")
                    print(f"export OUTPUT_DIR={os.path.abspath(output_dir)}/results")

    finally:
        # 카메라 닫고 창 제거
        zed.close()
        cv2.destroyAllWindows()
        print("ZED 카메라가 닫혔습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZED 카메라를 사용하여 SAM-6D용 데이터 캡처")
    parser.add_argument("--output_dir", type=str, help="출력 디렉토리 경로")
    parser.add_argument("--cad_path", type=str, help="CAD 모델 경로(.ply 파일)")
    args = parser.parse_args()
    
    main(args) 