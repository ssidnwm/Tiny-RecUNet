# Tiny-RecUnet

## Brain MRI Segmentation 프로젝트 파이프라인
### **디바이스 초기화**

- CUDA 사용 가능 여부를 확인해 디바이스를 선택합니다.
    - device = "cuda:0" if torch.cuda.is_available() else "cpu"
    - 선택된 디바이스를 로그로 출력합니다.

### **설정 불러오기**

- config.py에서 다음을 가져옵니다.
    - 실험 기본 설정:  `batch_size, epochs, lr, workers, image_size, aug_scale, aug_angle, weights, exp_name` 등
    - 모델 선택과 인자: `model_name, model_args`
- 모델 레지스트리: `models.model_dict`에서 `model_name` 키로 모델 클래스를 선택합니다.

### **데이터 로더**

- `loader_train`, `loader_valid` = `data_loaders(batch_size, workers, image_size, aug_scale, aug_angle)`.

### 데이터셋 구성(BrainSegmentationDataset)

- subset: "`train`", "`validation`", "`all`" 3가지 모드 지원
- 폴더 순회로 환자 단위(폴더 단위) 이미지를 읽고 3D 볼륨으로 구성
    - 이미지: `skimage.io.imread(filepath)` (컬러)
    - 마스크: `skimage.io.imread(filepath, as_gray=True)` (흑백)
    - 파일명 정렬은 슬라이스 인덱스 숫자를 기준으로 수행하여 이미지/마스크가 같은 순서로 정렬됨
    - 첫/마지막 슬라이스 제외: 중앙부 슬라이스만 사용해 엣지 노이즈를 줄임
    - `volumes[patient_id] = np.array(image_slices[1:-1])`
    - `masks[patient_id] = np.array(mask_slices[1:-1])`
- (volume, mask) 리스트 생성 후 전처리
    - `crop_sample`: 실제 유효 신호 영역을 감싸는 최소 박스로 잘라내기
    - `pad_sample`: 정사각형이 되도록 패딩
    - `resize_sample`: 구성된 image_size로 공간 해상도 변경
    - `normalize_volume`: 채널별로 intensity 정규화
- 랜덤 샘플링 확률 생성(양성 위주 + 10% 균등 스무딩)
    - 각 슬라이스 k에 대해 s[k] = 마스크 픽셀 합계
    - 확률 `p = (s + sum(s)*0.1/len(s)) / (sum(s)*1.1)`
    - 의미: 양성 슬라이스를 더 자주 뽑되, 음성도 일정 확률로 선택
- 마스크 채널 차원 추가
    - (Z,H,W) → (Z,H,W,1)로 바꿔 이후 (C,H,W)로 transpose하기 위함
- 전역 인덱스 맵 구성: 환자/슬라이스 → 단일 인덱스로 매핑
- **getitem**:
    - train에서는 확률 p에 따라 환자/슬라이스를 랜덤 샘플링
    - (H,W,C) → (C,H,W)로 transpose한 뒤 tensor로 반환
    - 마스크는 0~1 스케일로 정규화

### 데이터 로더 생성

- Train: shuffle=True, drop_last=True
- Valid: shuffle=False, drop_last=False
- num_workers=workers, worker_init_fn=worker_init 사용

### **모델 및 하이퍼파라미터**

- `ModelClass = model_dict[model_name]`
- `model = ModelClass(**model_args[model_name])`
- `model.to(device)`로 CUDA/CPU에 올림

### **손실/옵티마이저/로깅**

- Dice 손실: `dsc_loss = DiceLoss()` (모델 출력 확률에 맞춰 사용)
- 옵티마이저: `optim.Adam(model.parameters(), lr=lr)`
- 로깅용 리스트
    - `loss_train`, `loss_valid` (배치별 loss 누적)
    - `train_loss_history`, `valid_loss_history` (에포크 평균 저장)
- Best 모델 추적
    - `best_validation_dsc = 0.0`
    - 각 에포크 검증 후 평균 DSC가 최고면 체크포인트 저장

### **학습 루프(에포크 단위)**

- for epoch in range(epochs):
    - Train 단계
        - `model.train()`
        - 배치 반복:
            - 입력/정답을 device로 이동
            - `optimizer.zero_grad()`
            - forward → `loss = dsc_loss(y_pred, y_true)`
            - `loss.backward()`, `optimizer.step()`
            - `loss_train`에 배치 loss 기록
        - 에포크 종료 시 `train_loss_history`에 평균 훈련 손실 기록
    - Valid 단계
        - `model.eval()`
        - 배치 반복(그래디언트 비활성화)
            - forward만 수행
            - `loss_valid`에 배치 loss 기록
            - numpy로 변환해 예측/정답 리스트에 추가(후에 환자 단위로 합치기 위함)
        - 에포크 종료 시:
            - `valid_loss_history`에 평균 검증 손실 기록
            - `dsc_per_volume`로 환자 단위 DSC 계산 → `mean_dsc = np.mean(...)`
            - `mean_dsc`가 최댓값이면 torch.save(model.state_dict(), weights/<model_name>.pt)로 체크포인트 갱신

### **학습 종료 후(최종 평가/리포트)**

- 실험 결과 디렉토리 생성: `./result/{exp_name}`
- 훈련/검증 손실 곡선 저장: `loss_curve.png`
- 저장해둔 베스트 체크포인트 로드
    - `state_dict = torch.load(os.path.join(weights, f"{model_name}.pt"), weights_only=True)`
    - `model.load_state_dict(state_dict)`, `model.eval()`
- 최종 추론(검증 전체):
    - valid 로더를 다시 한 번 돌면서 입력/예측/정답을 모두 모음
    - `postprocess_per_volume`로 볼륨 재구성(환자 단위)
    - `dsc_distribution`/`plot_dsc`로 DSC 분포를 시각화 → dsc.png 저장
    - 각 환자-슬라이스별 결과 이미지 저장
        - `./result/{exp_name}/result_img/<patient>-<slice>.png`
        - FLAIR 채널(1)에 예측(빨강)/정답(초록) 윤곽선을 그려 저장
- 전체 경과 시간 출력
