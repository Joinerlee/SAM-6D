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
                # --- 추가: 이미지 데이터 검색 호출 --- 
                retrieve_status_img = zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU)
                if retrieve_status_img != sl.ERROR_CODE.SUCCESS:
                    print(f"[이미지 오류] 이미지 검색 실패: {retrieve_status_img}")
                    time.sleep(0.1)
                    continue
                # --- 추가 끝 ---
                
                # 이미지 데이터 가져오기 - get_data() 사용 및 철저한 검증
                try:
                    if not image_zed.is_init():
                        print("오류: 이미지 Mat 초기화 안됨")
                        continue
                        
                    # CPU 메모리로 데이터 가져오기
                    image_rgba = image_zed.get_data(sl.MEM.CPU)
                    
                    # 1. 유효성 검사 (초기)
                    if image_rgba is None or not isinstance(image_rgba, np.ndarray) or image_rgba.size == 0:
                        print("[이미지 오류] get_data() 결과가 유효하지 않습니다.")
                        continue
                        
                    # 2. 형태 및 채널 확인 (RGBA)
                    if len(image_rgba.shape) != 3 or image_rgba.shape[2] != 4:
                        print(f"[이미지 오류] 4채널 RGBA 이미지가 아닙니다: {image_rgba.shape}")
                        continue
                    
                    # 3. 데이터 타입 확인 (uint8)
                    if image_rgba.dtype != np.uint8:
                        print(f"[이미지 오류] 데이터 타입이 uint8이 아닙니다: {image_rgba.dtype}. 변환 시도...")
                        try:
                            image_rgba = image_rgba.astype(np.uint8)
                        except Exception as e:
                            print(f"[이미지 오류] uint8 변환 실패: {str(e)}")
                            continue
                            
                    # --- 수정: 명시적 복사본 생성 --- 
                    try:
                        # 완전히 새로운 배열 생성
                        image_rgba_copy = np.empty_like(image_rgba)
                        # 데이터 복사
                        np.copyto(image_rgba_copy, image_rgba)
                        print("[이미지 정보] get_data() 결과를 새 배열로 복사 완료.")
                    except Exception as copy_e:
                        print(f"[이미지 오류] NumPy 배열 복사 실패: {str(copy_e)}")
                        continue
                    # --- 수정 끝 ---
                            
                    # 4. RGBA to BGR 변환 (복사본 사용)
                    try:
                        # 복사된 배열을 사용하여 변환
                        image_rgb = cv2.cvtColor(image_rgba_copy, cv2.COLOR_RGBA2BGR)
                        print("[이미지 정보] RGBA->BGR 변환 성공 (복사본 사용).")
                    except Exception as cvt_e:
                        print(f"[이미지 오류] RGBA->BGR 변환 실패: {str(cvt_e)}")
                        continue

                    # 5. 메모리 연속성 보장 (변환 후에도 확인)
                    if not image_rgb.flags['C_CONTIGUOUS']:
                        print("[이미지 정보] BGR 변환 후 메모리가 연속적이지 않아 복사본 생성.")
                        image_rgb = np.ascontiguousarray(image_rgb)
                    
                    print("[이미지 정보] 최종 image_rgb 상태: 형태={image_rgb.shape}, 타입={image_rgb.dtype}, 연속성={image_rgb.flags['C_CONTIGUOUS']}")
                    
                except Exception as img_proc_e:
                    import traceback
                    print(f"[이미지 오류] 이미지 처리 중 예외 발생: {str(img_proc_e)}")
                    print(traceback.format_exc())
                    continue
                
                # --- 추가: 깊이 데이터 검색 호출 --- 
                retrieve_status_depth = zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH, sl.MEM.CPU)
                if retrieve_status_depth != sl.ERROR_CODE.SUCCESS:
                    print(f"[깊이 오류] 깊이 맵 검색 실패: {retrieve_status_depth}")
                    time.sleep(0.1)
                    continue
                # --- 추가 끝 ---
                
                # 깊이 데이터 가져오기 - get_data() 사용 및 철저한 검증
                try:
                    if not depth_zed.is_init():
                        print("오류: 깊이 Mat 초기화 안됨")
                        continue
                        
                    # CPU 메모리로 데이터 가져오기 (float32 예상)
                    depth_map = depth_zed.get_data(sl.MEM.CPU)
                    
                    # 1. 유효성 검사 (초기)
                    if depth_map is None or not isinstance(depth_map, np.ndarray) or depth_map.size == 0:
                        print("[깊이 오류] get_data() 결과가 유효하지 않습니다.")
                        continue
                        
                    # 2. 형태 확인 (2D)
                    if len(depth_map.shape) != 2:
                        print(f"[깊이 오류] 2차원 깊이 맵이 아닙니다: {depth_map.shape}")
                        continue
                        
                    # 3. 데이터 타입 확인 (float32)
                    if depth_map.dtype != np.float32:
                        print(f"[깊이 정보] 데이터 타입이 float32가 아닙니다: {depth_map.dtype}. 변환 시도...")
                        try:
                            depth_map = depth_map.astype(np.float32)
                        except Exception as e:
                            print(f"[깊이 오류] float32 변환 실패: {str(e)}")
                            continue
                    
                    # 4. 메모리 연속성 보장
                    if not depth_map.flags['C_CONTIGUOUS']:
                        print("[깊이 정보] 메모리가 연속적이지 않아 복사본 생성.")
                        depth_map = np.ascontiguousarray(depth_map)
                        
                    print(f"[깊이 정보] 최종 depth_map 상태: 형태={depth_map.shape}, 타입={depth_map.dtype}, 연속성={depth_map.flags['C_CONTIGUOUS']}")
                        
                except Exception as depth_proc_e:
                    import traceback
                    print(f"[깊이 오류] 깊이 처리 중 예외 발생: {str(depth_proc_e)}")
                    print(traceback.format_exc())
                    continue

                # 이미지 표시 처리 - imshow만 사용
                try:
                    # 최종 image_rgb 유효성 재확인
                    if image_rgb is None or not isinstance(image_rgb, np.ndarray) or image_rgb.dtype != np.uint8 or not image_rgb.flags['C_CONTIGUOUS']:
                         print(f"[표시 오류] imshow 직전 image_rgb 최종 검증 실패")
                         continue
                    
                    # imshow 시도
                    cv2.imshow("ZED Camera", image_rgb)
                    cv2.waitKey(1)
                    print("[표시 정보] imshow 호출 성공")
                        
                except Exception as e:
                    import traceback
                    print(f"[표시 오류] imshow 오류: {str(e)}")
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
                        hq_image_rgb = image_rgb.copy()
                        hq_depth_map = depth_map.copy()
                        
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