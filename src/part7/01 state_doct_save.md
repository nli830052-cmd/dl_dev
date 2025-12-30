## CNN 이미지 분류기 만들기 강의 정리

## 강의 개요

이 강의에서는 CNN(Convolutional Neural Network)을 이용한 이미지 분류기를 구현하는 방법을 배웁니다. 특히 MNIST 손글씨 데이터셋을 활용하여 0부터 9까지의 숫자를 분류하는 모델을 처음부터 끝까지 만드는 과정을 다룹니다.

## 핵심 개념 정리

### 1. 프로젝트 개요

* **목표** : MNIST 손글씨 숫자 이미지(0-9)를 분류하는 CNN 모델 구현
* **이미지 데이터** : 28x28 픽셀 크기의 흑백 이미지 (채널 수: 1)
* **분류 클래스** : 10개 클래스(0-9)의 다중 분류 문제

### 2. CNN 구조 이해

* **입력 데이터 형태** : [배치 크기, 1, 28, 28] - 흑백 이미지라서 채널이 1
* **출력 데이터 형태** : [배치 크기, 10] - 10개 클래스에 대한 예측 확률
* **주요 구성 요소** :
* 컨볼루션 층(Conv): 이미지 특징 추출
* ReLU 활성화 함수: 비선형성 추가
* 최대 풀링(MaxPooling): 특징맵 다운샘플링
* 플래튼(Flatten): 3D → 1D로 변환
* 선형 층(Linear): 최종 분류

### 3. 채널의 역할 이해

* **입력 채널(in_channel)** :
* 입력 이미지에서는 색상 정보 (흑백=1, 컬러=3)
* CNN 중간층에서는 이전 층의 출력 채널 수
* **출력 채널(out_channel)** :
* 데이터를 분해하는 역할
* 예: 1개 채널 → 10개로 분해 → 특징 추출

### 4. Softmax와 Cross Entropy Loss

* **Softmax 함수** :
* 출력값을 확률로 변환 (0~1 사이의 값, 합계 1)
* 다중 분류에 필요한 예측 확률 생성
* **Cross Entropy Loss** :
* 다중 분류에 적합한 손실 함수
* 파이토치에서는 내부적으로 Softmax 함수 포함됨

### 5. 학습 과정(Train)

1. 모델을 학습 모드로 설정(`model.train()`)
2. 데이터를 GPU로 이동
3. 순전파(forward) 수행
4. 손실(loss) 계산
5. 기울기 초기화(`optimizer.zero_grad()`)
6. 역전파 수행(기울기 계산)(`loss.backward()`)
7. 가중치 업데이트(`optimizer.step()`)

### 6. 평가 과정(Test)

1. 모델을 평가 모드로 설정(`model.eval()`)
2. 기울기 계산 비활성화(`with torch.no_grad():`)
3. 데이터를 GPU로 이동
4. 순전파(forward) 수행
5. 손실(loss) 계산
6. 정확도(accuracy) 계산
   * 예측값과 실제값 비교
   * 정확도 = (맞은 개수 / 전체 개수) × 100

## 파이토치 모델 저장 및 로드 실습 정리

## 강의개요

이번 강의는 파이토치에서 학습된 딥러닝 모델을 저장하고 다시 불러오는 다양한 방법을 다룹니다. 실제 프로젝트에서 학습된 모델을 저장하고 재사용하는 것은 매우 중요한 과정이며, 특히 오랜 시간이 소요되는 학습 결과를 보존하기 위해 필수적인 기술입니다.

## 핵심개념 정리

### 1. 모델 저장의 필요성

* 학습에 많은 시간과 자원이 소요됨
* 학습된 모델은 딥러닝 프로젝트의 가장 소중한 자산
* 저장된 모델은 추론, 공유, 재학습 등에 활용 가능

### 2. 파이토치의 모델 저장 방식 (3가지)

1. **state_dict 저장 방식** (권장 방식)
   * 모델의 학습 가능한 파라미터만 저장
   * 유연성, 이식성, 보안성 우수
   * 구조는 저장하지 않고 파라미터만 저장
2. **전체 모델 저장 방식**
   * 모델의 구조와 파라미터를 모두 저장
   * 코드 변경 시 로드에 실패할 가능성 있음
   * 보안 위험 존재 (pickle 사용)
3. **체크포인트 저장 방식**
   * 학습 중단 시점부터 정확히 재개하기 위한 모든 정보 저장
   * 에포크 번호, 모델 파라미터, 옵티마이저 상태, 손실값 등을 모두 저장
   * 최적의 모델만 선별적으로 저장 가능

### 3. state_dict 저장 및 로드 방법

* **저장** : `torch.save(model.state_dict(), path)`
* **로드** :

1. 동일 구조의 모델 객체 생성
2. `state_dict = torch.load(path, map_location='cpu')`
3. `model.load_state_dict(state_dict)`

### 4. 전체 모델 저장 및 로드 방법

* **저장** : `torch.save(model, path)`
* **로드** : `model = torch.load(path)`
* 간결하지만 보안 위험성이 있음 (pickle 모듈 사용)

## 강조사항

* 딥러닝에서는 학습이 오래 걸리므로 체크포인트 저장은 필수적
* 모델 학습 시 Early Stopping 보다 Dropout이나 BatchNormalization 같은 방법을 먼저 적용하는 것이 바람직함
* 학습 중 과적합 신호 확인 가능 (검증 손실이 증가하기 시작할 때)
* 모델 저장 시 파일명에 에포크, 손실값, 날짜 등 정보 포함하면 관리에 용이


## 강의요약: 딥러닝 과적합 방지 기법

## 강의개요

이 강의는 딥러닝 모델에서 발생하는 과적합(Overfitting) 문제와 이를 방지하기 위한 4가지 주요 기법에 대해 설명합니다. 과적합 발생 시 훈련 데이터에서는 좋은 성능을 보이지만 검증 데이터에서는 성능이 저하되는 문제를 해결하는 방법들을 코드 예제와 함께 소개합니다.

## 핵심개념 정리

### 1. 과적합(Overfitting) 문제

* 훈련 데이터에서는 성능이 좋지만 검증 데이터에서 성능이 저하되는 현상
* 모델이 훈련 데이터의 패턴을 과도하게 학습하여 일반화 능력이 떨어지게 됨
* 학습 곡선에서 검증 손실(validation loss)이 증가하기 시작하는 지점에서 과적합 발생

### 2. 과적합 방지 기법

#### 2.1 드롭아웃(Dropout)

* **개념** : 신경망의 일부 뉴런을 랜덤하게 비활성화하여 학습 진행
* **구현 방법** :
  **python**copy.label

<pre><div class="text-03-R scrollbar"><span class="token plain">self</span><span class="token punctuation">.</span><span class="token plain">dropout </span><span class="token operator">=</span><span class="token plain"> nn</span><span class="token punctuation">.</span><span class="token plain">Dropout</span><span class="token punctuation">(</span><span class="token plain">p</span><span class="token operator">=</span><span class="token number">0.5</span><span class="token punctuation">)</span><span class="token plain"></span><span class="token comment"># 50%의 뉴런을 비활성화</span></div></pre>

* **특징** :
* Linear 층 사이에 적용 (Convolution 층 사이에는 사용하지 않음)
* 학습 시에만 적용되고, 추론 시에는 모든 뉴런 사용
* 매 학습 반복마다 비활성화되는 뉴런이 랜덤하게 변경됨

#### 2.2 L2 규제화(Weight Decay)

* **개념** : 가중치가 너무 커지는 것을 방지하여 모델을 안정화
* **구현 방법** :
  **python**copy.label

<pre><div class="text-03-R scrollbar"><span class="token plain">optimizer </span><span class="token operator">=</span><span class="token plain"> torch</span><span class="token punctuation">.</span><span class="token plain">optim</span><span class="token punctuation">.</span><span class="token plain">Adam</span><span class="token punctuation">(</span><span class="token plain">model</span><span class="token punctuation">.</span><span class="token plain">parameters</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span><span class="token plain"> lr</span><span class="token operator">=</span><span class="token plain">learning_rate</span><span class="token punctuation">,</span><span class="token plain"> weight_decay</span><span class="token operator">=</span><span class="token number">1e-5</span><span class="token punctuation">)</span></div></pre>

* **특징** :
* 옵티마이저의 `weight_decay` 파라미터로 설정
* 작은 값(epsilon)을 사용하여 가중치의 크기를 제한

#### 2.3 학습률 스케줄러(Learning Rate Scheduler)

* **개념** : 학습 과정에서 학습률을 점진적으로 조정
* **구현 방법** :
  **python**copy.label

<pre><div class="text-03-R scrollbar"><span class="token plain">scheduler </span><span class="token operator">=</span><span class="token plain"> torch</span><span class="token punctuation">.</span><span class="token plain">optim</span><span class="token punctuation">.</span><span class="token plain">lr_scheduler</span><span class="token punctuation">.</span><span class="token plain">StepLR</span><span class="token punctuation">(</span><span class="token plain">optimizer</span><span class="token punctuation">,</span><span class="token plain"> step_size</span><span class="token operator">=</span><span class="token number">5</span><span class="token punctuation">,</span><span class="token plain"> gamma</span><span class="token operator">=</span><span class="token number">0.5</span><span class="token punctuation">)</span></div></pre>

* **특징** :
* 초기에는 큰 폭으로 학습하다가 점차 작은 폭으로 조정
* 5 에폭마다 학습률을 0.5배(절반)로 감소시킴
* 에폭마다 `scheduler.step()`을 호출하여 학습률 업데이트

#### 2.4 조기 종료(Early Stopping)

* **개념** : 검증 손실이 일정 기간 동안 개선되지 않으면 학습 중단
* **구현 방법** :
  **python**copy.label

<pre><div class="text-03-R scrollbar"><span class="token plain">patience </span><span class="token operator">=</span><span class="token plain"></span><span class="token number">5</span><span class="token plain"></span><span class="token comment"># 5번 동안 개선이 없으면 중단</span><span class="token plain"></span></div><div class="text-03-R scrollbar"><span class="token plain">patience_counter </span><span class="token operator">=</span><span class="token plain"></span><span class="token number">0</span><span class="token plain"></span></div><div class="text-03-R scrollbar"><span class="token plain">best_val_loss </span><span class="token operator">=</span><span class="token plain"></span><span class="token builtin">float</span><span class="token punctuation">(</span><span class="token string">'inf'</span><span class="token punctuation">)</span><span class="token plain"></span></div><div class="text-03-R scrollbar"><span class="token plain">
  </span></div><div class="text-03-R scrollbar"><span class="token plain"></span><span class="token comment"># 학습 루프 내</span><span class="token plain"></span></div><div class="text-03-R scrollbar"><span class="token plain"></span><span class="token keyword">if</span><span class="token plain"> val_loss </span><span class="token operator"><</span><span class="token plain"> best_val_loss</span><span class="token punctuation">:</span><span class="token plain"></span></div><div class="text-03-R scrollbar"><span class="token plain">    patience_counter </span><span class="token operator">=</span><span class="token plain"></span><span class="token number">0</span><span class="token plain"></span></div><div class="text-03-R scrollbar"><span class="token plain">    best_val_loss </span><span class="token operator">=</span><span class="token plain"> val_loss</span></div><div class="text-03-R scrollbar"><span class="token plain">    torch</span><span class="token punctuation">.</span><span class="token plain">save</span><span class="token punctuation">(</span><span class="token plain">model</span><span class="token punctuation">.</span><span class="token plain">state_dict</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span><span class="token plain"></span><span class="token string">'best.pth'</span><span class="token punctuation">)</span><span class="token plain"></span><span class="token comment"># 최적 모델 저장</span><span class="token plain"></span></div><div class="text-03-R scrollbar"><span class="token plain"></span><span class="token keyword">else</span><span class="token punctuation">:</span><span class="token plain"></span></div><div class="text-03-R scrollbar"><span class="token plain">    patience_counter </span><span class="token operator">+=</span><span class="token plain"></span><span class="token number">1</span><span class="token plain"></span></div><div class="text-03-R scrollbar"><span class="token plain"></span><span class="token keyword">if</span><span class="token plain"> patience_counter </span><span class="token operator">>=</span><span class="token plain"> patience</span><span class="token punctuation">:</span><span class="token plain"></span></div><div class="text-03-R scrollbar"><span class="token plain"></span><span class="token keyword">break</span><span class="token plain"></span><span class="token comment"># 학습 중단</span></div></pre>

* **특징** :
* PyTorch는 기본적으로 Early Stopping을 지원하지 않아 직접 구현 필요
* PyTorch Lightning에서는 기본 지원됨

## 사례/예시

* **MNIST 데이터셋 실험** : 구현된 모든 기법을 적용했을 때 99.18%의 높은 정확도 달성
* **학습 중단 사례** : 8번째 에폭에서 검증 손실이 0.46에서 0.54로 증가하고, 이후 5번 동안 개선되지 않아 13번째 에폭에서 학습 중단

## 강조사항

1. **PyTorch에서는 조기 종료 기능이 기본으로 제공되지 않음**
   * 직접 구현해야 하며, 최적 모델 저장(torch.save) 기능도 함께 구현 필요
2. **드롭아웃은 Linear 층 사이에만 적용해야 함**
   * Convolution 층 사이에 사용하려면 Dropout2d와 같은 특별한 드롭아웃 사용 필요
3. **딥러닝은 데이터가 많을수록 좋은 성능**
   * 조기 종료가 빨리 일어난다면, 데이터가 부족할 수 있음을 고려해야 함
   * 실제 현장 데이터는 MNIST처럼 깨끗하지 않으므로 더 많은 데이터 확보 필요
4. **네 가지 기법을 모두 적용**하여 과적합 문제를 종합적으로 해결하는 것이 효과적
