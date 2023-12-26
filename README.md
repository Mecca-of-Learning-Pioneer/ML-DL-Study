# ML-DL-Study
📘MLP에서 진행하는 ML, DL Study입니다.📘 

## Algorithm

### Machine Learning(ML) - Structured Data

> #### Supervised Learning(지도 학습)
>
> - Classification(분류) - Loss function : Cross-Entropy, Evaluation metrics : Accuracy (다른 것 훨씬 많음)
>   - k-NN
>     - 어떤 데이터에 대한 답을 구할 때, 주변의 k개의 가장 가까운 데이터를 보고 다수를 차지하는 것을 정답으로 판단
>   - Logistic Regression(기본 L2 Regularization)
>     - Binary Classification(이진 분류)
>       - feature를 가장 잘 나타낼 수 있는 선형 방정식을 양성 class에 대해서만 학습 -> 새로운 데이터를 넣으면, 선 위의 값을 반환(z) -> z값을 양성 class 확률로 변환(1에서 뺀 나머지는 음성 class 확률)(sigmoid function)
>     - Multiclass Classification(다중 분류)
>       - feature를 가장 잘 나타낼 수 있는 선형 방정식을 각 class별로 학습 -> 새로운 데이터를 넣으면, 선 위의 값을 반환(z) -> z값을 class별 확률로 변환(softmax function)
>   - Decision Tree(Normalization 필요X)
>     - 예/아니오에 대한 질문을 이어나가면서 정답을 찾아 학습 -> Leaf Node에서 가장 많은 class가 예측 class
>   - Ensemble Learning(대부분 Decesion Tree 기반)
>     - 더 좋은 예측 결과를 만들기 위해 여러 개의 model 훈련
>     - Random Forest, Extra Trees, Gradient Boosting, Histogram-based Gradient Boosting, (XGBoost) (Gradient Boosting 방법은 Gradient Descent를 이용한 것)
>
> - Regression(회귀) - Loss function : Mean Squared Error(MSE), Evaluation metrics : R-squared (다른 것 훨씬 많음)
>   - k-NN
>     - 어떤 데이터에 대한 답을 구할 때, 주변의 k개의 가장 가까운 데이터의 값을 평균낸 것을 정답으로 판단
>   - Linear Regression
>     - feature를 가장 잘 나타낼 수 있는 선형 방정식을 학습 -> 새로운 데이터를 넣으면, 선 위의 값을 반환(예측 값)
>     - Ridge Regression
>       - Linear Regression + L2 Regularization
>     - Lasso Regression
>       - Linear Regression + L1 Regularization
>   - Decision Tree(Normalization 필요X)
>     - 예/아니오에 대한 질문을 이어나가면서 정답을 찾아 학습 -> Leaf Node에 도달한 샘플의 target을 평균한 값이 예측값
>   - Ensemble Learning(대부분 Decesion Tree 기반)
>     - 더 좋은 예측 결과를 만들기 위해 여러 개의 model 훈련
>     - Random Forest, Extra Trees, Gradient Boosting, Histogram-based Gradient Boosting, (XGBoost) (Gradient Boosting 방법은 Gradient Descent를 이용한 것)

> #### Unsupervised Learning(비지도 학습)
>
> - Clustering(군집)
>   - k-Means
>     - cluster의 평균값(cluster center/centroid)을 랜덤으로 k개 정함 -> [각 샘플이 가장 가까운 centroid에 가서 cluster를 이룸 -> 다시 centroid를 구함] -> centroid에 변화가 없을 때까지 []구간 반복 -> k개의 cluster로 분류됨
>     - elbow 방법으로 최적의 k값을 찾을 수 있음
> 
> - Dimensionality Reduction(차원축소)
>   - Principal Component Analysis = PCA - Principal Component(주성분) : 데이터의 특징을 잘 표현하는 어떤 벡터
>     - 여러 주성분 중 일부 주성분만을 선택해서 데이터의 dimension(feature)를 줄임


> #### Reinforcement Learning(강화 학습)
> - RL Study에서 진행

#### Tip!
- 반드시 Normalization을 하고 Regularization 적용
- Model parameter : model이 feature에서 학습한 parameter
- Hyperparameter : 사람이 지정하는 parameter
- 점진적 학습 : Loss function 값을 점점 줄이는 방향으로 훈련하는 학습법
  - Gradient Descent
    - Stochastic Gradient Descent
    - Minibatch Gradient Descent
    - Batch Gradient Descent
- Validation Set : hyperparameter 튜닝을 위해 model을 평가할 때, test set를 사용하지 않기 위해 train set에서 다시 떼어 낸 data set
- Test Set는 model 완성 후 마지막에 한 번만 사용(여러번 사용 시, model을 test set에 맞추는 것이기 때문)
  - k-fold Cross Validtaion : train set를 k개의 fold로 나눈 다음 한 fold가 validation set의 역할, 나머지 fold는 model 훈련 -> k번 반복하여 얻은 모든 validation score를 평균냄
- Hyperparameter tuning with AutoML
  - Grid Search : hyperparameter 탐색(값의 목록 전달) + cross validation
  - Random Search : hyperparameter 탐색(값의 범위 전달) + cross validation
- Dimensionality Reduction을 하여 데이터의 dimension(feature)를 줄인 뒤, 다른 ML 알고리즘을 사용하여 훈련할 수 있음
- 데이터의 dimension(feature)을 3개 이하로 줄이면, 시각화하기 좋음(3D or 2D로 표현 가능)

### Deep Learning(DL) - Structured Data
> - DL의 목표 : loss function의 최소값 찾기 = Optimization(최적화)
> - ANN : Artificial Neural Network(인공 신경망)
> - DNN : Deep Neural Network(심층 신경망)
> - SLP : Single Layer Perceptron(단층 퍼셉트론)
> - MLP : Multi Layer Perceptron(다층 퍼셉트론)
> - Node : ANN을 구성하는 모든 feature를 node라고 부름

> - ANN ⊃ DNN
> - ANN = { SLP, {DNN} }
> - DNN = { MLP, CNN, RNN }

> - Input Layer(입력층)
> - Hidden Layer(은닉층) ⊃ FC(Fully Connected) Layer = Dense Layer(밀집층)
> - Output Layer(출력층)

> - Activation Function(활성화 함수) : 각 node의 선형방정식 계산 결과에 적용되는 함수
>   - sigmoid function(시그모이드 함수)
>   - softmax function(소프트맥스 함수)
>   - ReLU(렐루 함수)
>   - tanh(하이퍼볼릭 탄젠트 함수)

> - Optimizer : 최소의 loss값을 찾는 최적화 알고리즘
>   - SGD
>   - Adaptive Learning Rate(적응적 학습률) 사용하는 optimizer - 모델이 최적점에 가까이 갈수록 학습률을 낮출 수 있음
>     - Adagrad
>     - RMSprop
>     - Adam : Momentum optimization + RMSprop

> - Dropout : hidden layer에 있는 일부 node를 끄고 훈련시키는 것 - overfitting 방지

> - 일반 Data : 순서에 의미가 없는 data, 예) Image
> - Sequential Data(순차 데이터) : 순서에 의미가 있는 data, 예) Text, Time Series(시계열)(일정한 시간 간격으로 기록된 데이터)
>   - sequence : 하나의 샘플
>   - Text Data
>     - token : 분리된 단어
>     - 어휘 사전 : train set에서 고유한 단어를 뽑아 만든 목록
>     - 단어마다 고유한 정수를 부여해 숫자 데이터로 바꿈(0 : padding, 1 : 문장의 시작, 2 : 어휘 사전에 없는 toekn)
>     - 정수값 사이에는 어떠한 관계도 없음 - One-hot encoding, Word embedding 이용

> - CNN(Convolution Neural Network)(합성곱 신경망) - image data 처리에 특화되어 있음
>   - convolution layer을 가지고 있는 NN
>   - filter = kernel => convolution layer에서 feature map 생성
>   - same padding : convolution 과정을 거치고도 output 크기를 input과 동일하게 하기 위함
>   - stride : convolution 연산 과정에서 filter의 이동 크기
>   - pooling : convolution layer에서 생성된 feature map의 가로세로 크기를 줄임
>     - max pooling : filter를 찍은 영역에서 가장 큰 값 고르기
>     - average pooling : filter를 찍은 영역에서 평균값 계산하기

> - RNN(Recurrent Neural Network)(순환 신경망) - sequential data 처리에 특화되어 있음
>   - recurrent layer을 가지고 있는 NN
>   - rucurrent layer = cell
>   - cell의 출력 = hidden state
>   - timestep : 샘플을 처리하는 한 단계, text data - 1개의 token이 하나의 timestep
>   - NLP(Natural Language Processing)(자연어 처리)에서 이용 - 음성 인식, 기계 번역, 감성 분석 등
>   - 주요 Model
>     - LSTM(Long Short-Term Memory)
>     - GRU(Gated Recurrent Unit)

#### Tip!
- 이미지 픽셀은 0~255 사이의 정수값을 가짐 -> 255로 나누어 0~1 사이의 값으로 normalization
- DL에서는 cross validation 대신 validation set을 별도로 덜어내어 사용
- 모든 hidden layer와 output layer에는 bias(절편)과 activation function이 있음(단, Regression의 경우에는 output layer에 activation function이 없음)
- Binary Classification을 할 때 output layer에서는 sigmoid function 사용
- Multiclass Classification을 할 때 output layer에서는 softmax function 사용
- 보통 convolution layer와 pooling layer은 거의 항상 함께 사용
