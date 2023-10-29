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
>     - Binary Classification
>       - feature를 가장 잘 나타낼 수 있는 선형 방정식을 양성 class에 대해서만 학습 -> 새로운 데이터를 넣으면, 선 위의 값을 반환(z) -> z값을 양성 class 확률로 변환(1에서 뺀 나머지는 음성 class 확률)(sigmoid function)
>     - Multiclass Classification
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

> #### Unsupervised Learning


> #### Reinforcement Learning
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
