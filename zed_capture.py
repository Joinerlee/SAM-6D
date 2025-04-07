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
            grab_status = zed.grab(runtime_parameters)
            if grab_status != sl.ERROR_CODE.SUCCESS:
                print(f"프레임 획득 실패: {grab_status}. 다시 시도합니다...")
                time.sleep(0.1)
                continue
                
            try:
                # 1. 이미지 데이터 가져오기 (기본 해상도로)
                retrieve_status = zed.retrieve_image(image_zed, sl.VIEW.LEFT)
                if retrieve_status != sl.ERROR_CODE.SUCCESS:
                    print(f"이미지 검색 실패: {retrieve_status}, 다시 시도합니다...")
                    time.sleep(0.1)
                    continue
                
                # 디버깅: 이미지 매트 상태 확인
                if not image_zed.is_init():
                    print("오류: 이미지 매트가 초기화되지 않았습니다")
                    time.sleep(0.1)
                    continue
                    
                # 첫 번째 성공한 프레임에서만 매트 정보 출력
                if "first_frame_info" not in locals():
                    print(f"이미지 매트 정보: 너비={image_zed.get_width()}, 높이={image_zed.get_height()}, 채널={image_zed.get_channels()}")
                    first_frame_info = True
                
                # 이미지 데이터 가져오기 - ZED 이미지를 OpenCV로 변환하는 방식 개선
                try:
                    # 이미지 직접 추출 - 속도/품질 설정
                    downsample = 4  # 기본값
                    if args.fast_mode:
                        downsample = 8  # 빠른 미리보기용
                    elif args.high_quality:
                        downsample = 1  # 고품질 저장용
                    
                    print(f"이미지 직접 변환 시작... (다운샘플링: {downsample}x)")
                    # 이미지 직접 추출 함수 수정
                    def zed_to_opencv_image(zed_mat, downsample=4):
                        try:
                            # 이미지 크기 정보 얻기
                            width = zed_mat.get_width()
                            height = zed_mat.get_height()
                            
                            # 빈 OpenCV BGR 이미지 생성
                            opencv_image = np.zeros((height, width, 3), dtype=np.uint8)
                            
                            # 진행 표시 간격
                            progress_step = height // 10
                            
                            # 픽셀별 접근은 느리므로 성능 향상을 위해 다운샘플링
                            # 다운샘플링된 이미지에서 픽셀 가져오기
                            for y in range(0, height, downsample):
                                # 진행 상황 표시
                                if y % progress_step == 0:
                                    print(f"이미지 변환 중: {y/height*100:.1f}% 완료...")
                                
                                for x in range(0, width, downsample):
                                    # 각 픽셀의 컬러 값 가져오기
                                    try:
                                        pixel = zed_mat.get_value(x, y)
                                        
                                        # 유효한 픽셀 범위 계산 (오버플로우 방지)
                                        y_end = min(y+downsample, height)
                                        x_end = min(x+downsample, width)
                                        
                                        # BGR 순서로 저장 (OpenCV는 BGR 형식 사용)
                                        opencv_image[y:y_end, x:x_end, 0] = pixel[2]  # B
                                        opencv_image[y:y_end, x:x_end, 1] = pixel[1]  # G
                                        opencv_image[y:y_end, x:x_end, 2] = pixel[0]  # R
                                    except Exception as e:
                                        # 개별 픽셀 처리 오류 무시하고 계속 진행
                                        continue
                            
                            # 첫 번째 변환에만 표시
                            print("이미지 변환 완료!")
                            
                            # 확실히 연속된 메모리를 사용하는 배열로 변환
                            opencv_image = np.ascontiguousarray(opencv_image)
                            
                            return opencv_image
                        except Exception as e:
                            print(f"이미지 변환 오류: {str(e)}")
                            # 오류 시 검은색 이미지 반환
                            return np.zeros((height, width, 3), dtype=np.uint8)
                    
                    # 이미지 직접 추출 및 검증
                    image_rgb = zed_to_opencv_image(image_zed, downsample)
                    
                    # 반환된 이미지 검증
                    if image_rgb is None or not isinstance(image_rgb, np.ndarray) or image_rgb.size == 0:
                        print("오류: 유효하지 않은 이미지가 생성되었습니다. 검은색 이미지로 대체합니다.")
                        image_rgb = np.zeros((img_height, img_width, 3), dtype=np.uint8)
                    elif not image_rgb.flags['C_CONTIGUOUS']:
                        print("경고: 이미지가 메모리에 연속적으로 저장되어 있지 않습니다. 연속 배열로 변환합니다.")
                        image_rgb = np.ascontiguousarray(image_rgb)
                    
                    print("이미지 변환 성공!")
                    
                    # 첫 번째 프레임에서만 형식 정보 출력
                    if "first_image_format" not in locals():
                        print(f"직접 생성된 이미지 데이터 형식: 형태={image_rgb.shape}, 타입={image_rgb.dtype}")
                        first_image_format = True
                    
                    # 데이터가 유효한지 확인
                    if image_rgb is None:
                        print("오류: 이미지 데이터가 None입니다")
                        time.sleep(0.1)
                        continue
                
                except Exception as e:
                    import traceback
                    print(f"이미지 데이터 처리 오류: {str(e)}")
                    print(f"오류 상세정보: {traceback.format_exc()}")
                    time.sleep(0.1)
                    continue
                
                # 2. 깊이 맵 가져오기 (기본 해상도로)
                retrieve_status = zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)
                if retrieve_status != sl.ERROR_CODE.SUCCESS:
                    print(f"깊이 맵 검색 실패: {retrieve_status}, 다시 시도합니다...")
                    time.sleep(0.1)
                    continue
                
                # 디버깅: 깊이 매트 상태 확인
                if not depth_zed.is_init():
                    print("오류: 깊이 매트가 초기화되지 않았습니다")
                    time.sleep(0.1)
                    continue
                    
                # 첫 번째 성공한 프레임에서만 매트 정보 출력
                if "first_depth_info" not in locals():
                    print(f"깊이 매트 정보: 너비={depth_zed.get_width()}, 높이={depth_zed.get_height()}, 채널={depth_zed.get_channels()}")
                    first_depth_info = True
                
                # 깊이 데이터 가져오기 - 직접 픽셀 단위로 접근하는 방식으로 수정
                try:
                    # 깊이 맵 메모리에 직접 접근 대신 다른 방식으로 접근
                    if "first_debug_depth" not in locals():
                        print("깊이 데이터 추출 시도 중...")
                        first_debug_depth = True
                    
                    # 깊이 맵 직접 메모리 복사 방식
                    width = depth_zed.get_width()
                    height = depth_zed.get_height()
                    
                    # 깊이 맵 직접 복사 함수
                    def zed_to_opencv_depth(depth_mat, downsample=4):
                        try:
                            # 빈 깊이 맵 생성
                            opencv_depth = np.zeros((height, width), dtype=np.float32)
                            
                            # 진행 표시 간격
                            progress_step = height // 10
                            
                            # 픽셀별 접근은 느리므로 성능 향상을 위해 다운샘플링
                            for y in range(0, height, downsample):
                                # 진행 상황 표시
                                if y % progress_step == 0:
                                    print(f"깊이 맵 변환 중: {y/height*100:.1f}% 완료...")
                                    
                                for x in range(0, width, downsample):
                                    try:
                                        # 각 픽셀의 깊이 값 가져오기
                                        depth_value = depth_mat.get_value(x, y)
                                        
                                        # 유효한 픽셀 범위 계산 (오버플로우 방지)
                                        y_end = min(y+downsample, height)
                                        x_end = min(x+downsample, width)
                                        
                                        # 유효한 깊이값만 사용
                                        if depth_value[0] < 0:  # 무효한 깊이 값
                                            opencv_depth[y:y_end, x:x_end] = 0.0
                                        else:
                                            opencv_depth[y:y_end, x:x_end] = depth_value[0]  # 밀리미터 단위 깊이
                                    except Exception as e:
                                        # 개별 픽셀 처리 오류 무시하고 계속 진행
                                        continue
                                        
                            # 변환 완료 메시지
                            print("깊이 맵 변환 완료!")
                            
                            # 확실히 연속된 메모리를 사용하는 배열로 변환
                            opencv_depth = np.ascontiguousarray(opencv_depth)
                                
                            return opencv_depth
                        except Exception as e:
                            print(f"깊이 맵 변환 오류: {str(e)}")
                            # 오류 시 영행렬 반환
                            return np.zeros((height, width), dtype=np.float32)
                    
                    # 깊이 맵 직접 추출 - 속도/품질 설정
                    print(f"깊이 맵 직접 변환 시작... (다운샘플링: {downsample}x)")
                    # 깊이 맵 직접 추출
                    depth_map = zed_to_opencv_depth(depth_zed, downsample)
                    
                    # 반환된 깊이 맵 검증
                    if depth_map is None or not isinstance(depth_map, np.ndarray) or depth_map.size == 0:
                        print("오류: 유효하지 않은 깊이 맵이 생성되었습니다. 빈 깊이 맵으로 대체합니다.")
                        depth_map = np.zeros((img_height, img_width), dtype=np.float32)
                    elif not depth_map.flags['C_CONTIGUOUS']:
                        print("경고: 깊이 맵이 메모리에 연속적으로 저장되어 있지 않습니다. 연속 배열로 변환합니다.")
                        depth_map = np.ascontiguousarray(depth_map)
                    
                    print("깊이 맵 변환 성공!")
                    
                    # 첫 번째 프레임에서만 형식 정보 출력
                    if "first_depth_format" not in locals():
                        print(f"직접 생성된 깊이 데이터 형식: 형태={depth_map.shape}, 타입={depth_map.dtype}")
                        if len(depth_map.shape) > 0:
                            depth_min = np.min(depth_map)
                            depth_max = np.max(depth_map)
                            print(f"깊이 데이터 범위: 최소={depth_min}, 최대={depth_max}")
                        first_depth_format = True
                    
                    # 데이터가 유효한지 확인
                    if depth_map is None:
                        print("오류: 깊이 데이터가 None입니다")
                        time.sleep(0.1)
                        continue
                    
                except Exception as e:
                    import traceback
                    print(f"깊이 데이터 처리 오류: {str(e)}")
                    print(f"오류 상세정보: {traceback.format_exc()}")
                    time.sleep(0.1)
                    continue
                
                # 이미지와 깊이 데이터를 가져온 후 디버그 정보 출력
                if "first_debug_info" not in locals():
                    if image_rgb is not None:
                        print(f"이미지: 타입={type(image_rgb)}, 형태={image_rgb.shape}, 데이터타입={image_rgb.dtype}")
                    if depth_map is not None:
                        print(f"깊이 맵: 타입={type(depth_map)}, 형태={depth_map.shape}, 데이터타입={depth_map.dtype}")
                    if image_rgb is not None and depth_map is not None:
                        print(f"첫 번째 프레임 처리 성공!")
                        first_debug_info = True
                
                # 이미지 형태 확인 (디버깅용)
                if "first_frame" not in locals():
                    print(f"원본 이미지 형태: {image_rgb.shape}, 타입: {image_rgb.dtype}")
                    print(f"원본 깊이 맵 형태: {depth_map.shape}, 타입: {depth_map.dtype}")
                    first_frame = False
                
                # 이미지 표시 처리 - 최종 안전성 강화
                try:
                    # 1. image_rgb 유효성 검사 (초기)
                    if image_rgb is None or not isinstance(image_rgb, np.ndarray) or image_rgb.size == 0:
                        print("[표시 오류] 초기 image_rgb가 유효하지 않습니다.")
                        continue
                    
                    # 2. 필수 조건 확인 (형태, 채널, 타입)
                    if len(image_rgb.shape) != 3 or image_rgb.shape[2] != 3:
                        print(f"[표시 오류] 3채널 이미지가 아닙니다: {image_rgb.shape}")
                        continue
                    if image_rgb.dtype != np.uint8:
                        print(f"[표시 오류] 이미지 데이터 타입이 uint8이 아닙니다: {image_rgb.dtype}. 변환 시도...")
                        try:
                            image_rgb = image_rgb.astype(np.uint8)
                        except Exception as e:
                            print(f"[표시 오류] uint8 변환 실패: {str(e)}")
                            continue

                    # 3. 메모리 연속성 보장
                    if not image_rgb.flags['C_CONTIGUOUS']:
                        print("[표시 정보] 이미지가 메모리에 연속적이지 않아 복사본을 생성합니다.")
                        image_rgb = np.ascontiguousarray(image_rgb)
                        
                    # 4. putText 실행 전 최종 검사 (image_rgb 직접 검사)
                    if not isinstance(image_rgb, np.ndarray) or image_rgb.dtype != np.uint8 or not image_rgb.flags['C_CONTIGUOUS']:
                        print(f"[표시 오류] putText 직전 image_rgb 검증 실패: 타입={type(image_rgb)}, dtype={image_rgb.dtype}, 연속성={image_rgb.flags['C_CONTIGUOUS']}")
                        continue # 이 프레임은 건너뜀
                        
                    print(f"[표시 정보] putText 직전 image_rgb 상태: 형태={image_rgb.shape}, 타입={image_rgb.dtype}, 연속성={image_rgb.flags['C_CONTIGUOUS']}")

                    # 5. 텍스트 추가 시도 (이제 image_rgb 원본에 적용)
                    try:
                        # 여기서 image_rgb에 직접 텍스트를 그리도록 시도
                        label = f"ZED {img_width}x{img_height}"
                        font_face = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.7
                        color = (0, 255, 0)  # BGR 형식 (녹색)
                        thickness = 2
                        # putText를 image_rgb에 직접 적용 - 별도 try 블록
                        try:
                            cv2.putText(image_rgb, label, (10, 30), font_face, font_scale, color, thickness)
                            print("[표시 정보] putText 호출 성공!")
                        except Exception as text_e:
                            print(f"[표시 오류] putText 함수 자체에서 오류 발생: {str(text_e)}")
                            # 오류 발생 시 텍스트 없이 진행

                        # 6. 이미지 표시 시도 (텍스트 추가 여부와 관계없이 진행)
                        cv2.imshow("ZED Camera", image_rgb)
                        cv2.waitKey(1)
                        # 성공 메시지는 디버깅을 위해 유지
                        print(f"[표시 정보] imshow 호출 성공: 형태={image_rgb.shape}, 타입={image_rgb.dtype}")
                        
                    except Exception as e:
                        import traceback
                        print(f"[표시 오류] 텍스트 추가 또는 imshow 오류: {str(e)}")
                        print(traceback.format_exc())
                
                except Exception as e:
                    import traceback
                    print(f"[표시 오류] 전체 디스플레이 처리 블록 오류: {str(e)}")
                    print(traceback.format_exc())

                # --- 키 처리 ---
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 종료
                    print("종료 중...")
                    break
                elif key == ord('s'):  # 데이터 저장
                    print("\n고품질 이미지 캡처 시작...")
                    
                    # 고품질 이미지 캡처 준비
                    try:
                        # 고품질 이미지와 깊이 맵 생성 (다운샘플링 없음)
                        print("고품질 이미지 변환 중...")
                        hq_image_rgb = zed_to_opencv_image(image_zed, downsample=1)
                        
                        print("고품질 깊이 맵 변환 중...")
                        hq_depth_map = zed_to_opencv_depth(depth_zed, downsample=1)
                        
                        # 중앙 640x480 영역 추출
                        h, w = hq_image_rgb.shape[:2]
                        start_x = max(0, (w - 640) // 2)
                        start_y = max(0, (h - 480) // 2)
                        end_x = min(w, start_x + 640)
                        end_y = min(h, start_y + 480)
                        
                        # 안전하게 크롭
                        crop_w = end_x - start_x
                        crop_h = end_y - start_y
                        
                        # 이미지 크롭
                        if crop_w == 640 and crop_h == 480:
                            # 정확히 크기가 맞으면 직접 크롭
                            sam6d_color = hq_image_rgb[start_y:end_y, start_x:end_x].copy()
                            sam6d_depth = hq_depth_map[start_y:end_y, start_x:end_x].copy()
                        else:
                            # 크기가 다르면 패딩 처리
                            sam6d_color = np.zeros((480, 640, 3), dtype=np.uint8)
                            sam6d_depth = np.zeros((480, 640), dtype=np.float32)
                            
                            # 중앙에 배치
                            offset_y = (480 - crop_h) // 2
                            offset_x = (640 - crop_w) // 2
                            
                            if crop_h > 0 and crop_w > 0:
                                sam6d_color[offset_y:offset_y+crop_h, offset_x:offset_x+crop_w] = hq_image_rgb[start_y:end_y, start_x:end_x].copy()
                                sam6d_depth[offset_y:offset_y+crop_h, offset_x:offset_x+crop_w] = hq_depth_map[start_y:end_y, start_x:end_x].copy()
                        
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
                        print(f"RGB 이미지 저장 중... 형태: {sam6d_color.shape}")
                        cv2.imwrite(rgb_path, sam6d_color)
                        
                        # 16비트 PNG로 깊이 맵 저장 (SAM-6D와 호환되도록)
                        print(f"깊이 맵 저장 중... 형태: {sam6d_depth.shape}")
                        cv2.imwrite(depth_path, sam6d_depth.astype(np.uint16))
                        
                        # 카메라 정보 저장
                        with open(camera_path, "w") as f:
                            json.dump(intrinsics_dict, f, indent=4)
                        
                        print(f"SAM-6D용 데이터 저장 완료: {output_dir}")
                        
                        # CAD 모델 경로 출력
                        cad_path_info = args.cad_path if args.cad_path else "/home/recl3090/SAM-6D/TAN.ply (기본값)"
                        print(f"CAD 모델 경로: {cad_path_info}")
                        
                        # SAM-6D 실행 환경변수 예시
                        used_cad_path = args.cad_path if args.cad_path else "/home/recl3090/SAM-6D/TAN.ply"
                        print("\nSAM-6D 실행 환경변수 예시:")
                        print(f"export CAD_PATH={used_cad_path}")
                        print(f"export RGB_PATH={os.path.abspath(rgb_path)}")
                        print(f"export DEPTH_PATH={os.path.abspath(depth_path)}")
                        print(f"export CAMERA_PATH={os.path.abspath(camera_path)}")
                        print(f"export OUTPUT_DIR={os.path.abspath(output_dir)}/results")
                    except Exception as e:
                        import traceback
                        print(f"저장 오류: {str(e)}")
                        print(f"오류 상세정보: {traceback.format_exc()}")
                
            except Exception as e:
                import traceback
                print(f"처리 오류: {str(e)}")
                print(f"오류 상세정보: {traceback.format_exc()}")
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
    parser.add_argument("--fast_mode", action="store_true", help="빠른 모드 사용 (다운샘플링 8x, 미리보기 전용)")
    parser.add_argument("--high_quality", action="store_true", help="고품질 모드 사용 (다운샘플링 없음, 저장시 권장)")
    args = parser.parse_args()
    
    main(args) 