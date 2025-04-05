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
    init_params.camera_resolution = sl.RESOLUTION.HD720  # 해상도 설정 (1280x720)
    init_params.camera_fps = 30  # 프레임 속도 설정

    # 카메라 열기
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"ZED 카메라 열기 실패: {repr(err)}. 종료합니다...")
        zed.close()
        exit()

    # 이미지와 Depth 데이터를 담을 sl.Mat 객체 생성
    image_sl = sl.Mat()
    depth_sl = sl.Mat()
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
    
    # 원본 해상도 정보
    img_width = zed.get_camera_information().camera_configuration.resolution.width
    img_height = zed.get_camera_information().camera_configuration.resolution.height
    print(f"원본 해상도: {img_width}x{img_height}")
    
    # SAM-6D 대상 해상도
    target_width, target_height = 640, 480
    print(f"변환 대상 해상도: {target_width}x{target_height}")

    # 리사이징 비율 계산
    width_ratio = target_width / img_width
    height_ratio = target_height / img_height
    
    # 카메라 내부 파라미터 조정
    target_cam_k = [
        cam_k[0] * width_ratio, cam_k[1], cam_k[2] * width_ratio,
        cam_k[3], cam_k[4] * height_ratio, cam_k[5] * height_ratio,
        cam_k[6], cam_k[7], cam_k[8]
    ]
    target_intrinsics_dict = {"cam_K": target_cam_k}
    
    print("\n's' 키를 눌러 데이터를 저장하고, 'q' 또는 ESC 키를 눌러 종료하세요.")

    try:
        while True:
            # 새로운 프레임 잡기
            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                # 왼쪽 컬러 이미지 검색
                zed.retrieve_image(image_sl, sl.VIEW.LEFT, sl.MEM.CPU, (img_width, img_height))
                # Depth 맵 검색
                zed.retrieve_measure(depth_sl, sl.MEASURE.DEPTH, sl.MEM.CPU, (img_width, img_height))

                # NumPy 배열로 변환
                color_image_rgba = image_sl.get_data()
                depth_map = depth_sl.get_data()  # Depth 단위: MILLIMETERS

                # RGBA를 BGR로 변환 (OpenCV 형식)
                color_image_bgr = cv2.cvtColor(color_image_rgba, cv2.COLOR_RGBA2BGR)

                # Depth 맵의 NaN/Inf 값 처리 (0으로 대체)
                depth_map[np.isnan(depth_map)] = 0
                depth_map[np.isinf(depth_map)] = 0
                
                # 원본 이미지 표시용 리사이즈 (화면에 맞게)
                display_size = (640, 360)  # 16:9 비율 유지
                color_display = cv2.resize(color_image_bgr, display_size)
                
                # 표시용 Depth 이미지 생성
                depth_display = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
                depth_display = cv2.resize(depth_colormap, display_size)
                
                # SAM-6D 용으로 변환된 이미지 미리보기 (640x480으로 리사이즈)
                resized_color = cv2.resize(color_image_bgr, (target_width, target_height))
                resized_depth = cv2.resize(depth_map, (target_width, target_height))
                
                # 변환된 리사이즈 이미지 표시용 처리
                resized_depth_display = cv2.normalize(resized_depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                resized_depth_colormap = cv2.applyColorMap(resized_depth_display, cv2.COLORMAP_JET)
                
                # 이미지 두 줄로 표시 (원본 위, 변환 아래)
                top_row = np.hstack((color_display, depth_display))
                bottom_row = np.hstack((
                    cv2.resize(resized_color, display_size),
                    cv2.resize(resized_depth_colormap, display_size)
                ))
                display_image = np.vstack((top_row, bottom_row))
                
                # 설명 텍스트 추가
                cv2.putText(display_image, "Original (1280x720)", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(display_image, "SAM-6D (640x480)", (10, 390), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # 이미지 보여주기
                cv2.imshow("ZED 카메라 | 원본(위) | SAM-6D용 변환(아래)", display_image)

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
                    
                    # 원본 이미지 저장 (원하는 경우)
                    if args.save_original:
                        orig_dir = os.path.join(output_dir, "original")
                        os.makedirs(orig_dir, exist_ok=True)
                        orig_rgb_path = os.path.join(orig_dir, "rgb_original.png")
                        orig_depth_path = os.path.join(orig_dir, "depth_original.png")
                        orig_camera_path = os.path.join(orig_dir, "camera_original.json")
                        
                        cv2.imwrite(orig_rgb_path, color_image_bgr)
                        cv2.imwrite(orig_depth_path, depth_map.astype(np.uint16))
                        with open(orig_camera_path, "w") as f:
                            json.dump(intrinsics_dict, f, indent=4)
                        print(f"원본 이미지 저장 완료: {orig_dir}")

                    # SAM-6D용 변환 이미지 저장 (640x480)
                    cv2.imwrite(rgb_path, resized_color)
                    cv2.imwrite(depth_path, resized_depth.astype(np.uint16))
                    with open(camera_path, "w") as f:
                        json.dump(target_intrinsics_dict, f, indent=4)
                    
                    print(f"SAM-6D용 변환 데이터 저장 완료: {output_dir}")
                    
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
    parser.add_argument("--save_original", action="store_true", help="원본 해상도 이미지도 저장")
    args = parser.parse_args()
    
    main(args) 