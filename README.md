# Tiny-RecUnet

## Brain MRI Segmentation 프로젝트 파이프라인
### 1. 프로젝트 구조
- `main.py`: 실험 실행, 학습/검증 루프, 결과 저장

- `config.py`: 실험 설정(모델명, 하이퍼파라미터 등)

- `dataset.py`: 데이터셋 로딩, 전처리, DataLoader 정의

- `models`: 다양한 모델(UNet, ViT 등) 구현 및 선택

- `utils.py`: 손실함수(DiceLoss), 평가, 시각화, 로깅 등 유틸리티 함수

### 2. 전체 프로세스
1) 설정 및 모델 선택
   
- `config.py`에서 실험 파라미터(모델명, batch size, epoch 등) 지정
- `__init__.py`의 model_dict를 통해 원하는 모델 선택

2) 데이터 준비
- BrainSegmentationDataset에서 데이터 로딩 및 전처리(정규화, 크롭, 패딩, 리사이즈 등)
- data_loaders 함수로 train/valid DataLoader 생성

3) 학습 및 검증
- main.py에서 모델 인스턴스화 및 device(GPU/CPU) 설정
- 학습 루프: train/valid phase로 나누어 진행
- train: 모델 학습, loss 계산 및 optimizer step
- valid: 모델 평가, loss 및 Dice Similarity Coefficient(DSC) 계산
- 각 epoch마다 loss, val_loss, val_dsc 로깅 및 최적 모델 저장

4) 결과 분석 및 시각화
- 최적 모델로 validation 데이터 전체 예측
- 후처리(postprocess_per_volume)로 환자별 볼륨 결과 생성
- DSC 분포 시각화 및 저장
- 각 환자별 slice segmentation 결과 이미지 저장

5) 파일/폴더 관리
- 결과 이미지(result_img), loss curve(loss_curve_...png), DSC plot(dsc.png) 등 자동 저장
- 경로 및 폴더 생성 자동화

### 3. 주요 특징 및 장점
- 실험 설정 및 모델 선택이 config 파일과 model_dict로 매우 유연함
- 데이터 전처리, 학습, 평가, 시각화가 모듈화되어 유지보수 및 확장 용이
- GPU/CPU 자동 선택, 로깅 및 결과 저장 자동화


### 4. 사용 방법
- `config.py`에서 실험 파라미터와 모델명 지정
- `main.py` 실행 (python main.py)
- 결과는 로그 파일, 이미지, 그래프 등으로 자동 저장됨
