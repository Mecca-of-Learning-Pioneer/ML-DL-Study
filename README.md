# ML-DL-Study
📘MLP에서 진행하는 ML, DL Study입니다.📘 

## Algorithm

### Supervised Learning(지도 학습)
- Classification(분류) - 주로 Cross-Entropy Loss function을 사용
  - k-NN
    - 어떤 데이터에 대한 답을 구할 때, 주변의 k개의 가장 가까운 데이터를 보고 다수를 차지하는 것을 정답으로 판단
  - Logistic Regression(기본 L2 Regularization)
    - feature를 가장 잘 나타낼 수 있는 선형 방정식을 각 class별로 학습 -> 새로운 데이터를 넣으면, 선 위의 값을 반환(z) -> z값을 class별 확률로 변환(Binary Classification : sigmoid function, Multiclass Classification : softmax function)
      (Binary Classification의 경우 양성 class의 선형 방정식만 학습)

- Regression(회귀) - 주로 Mean Squared Error(MSE)를 Loss function으로 사용
  - k-NN
    - 어떤 데이터에 대한 답을 구할 때, 주변의 k개의 가장 가까운 데이터의 값을 평균낸 것을 정답으로 판단
  - Linear Regression
    - feature를 가장 잘 나타낼 수 있는 선형 방정식을 학습 -> 새로운 데이터를 넣으면, 선 위의 값을 반환
    - Ridge Regression
      - Linear Regression + L2 Regularization
    - Lasso Regression
      - Linear Regression + L1 Regularization

### Unsupervised Learning


### Reinforcement Learning
- RL Study에서 진행

#### Tip!
- 반드시 Normalization을 하고 Regularization 적용
