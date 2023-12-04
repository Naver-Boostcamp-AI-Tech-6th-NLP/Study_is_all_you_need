# GloVe의 Embedding 과정

## Reference
[고려대학교 DSBA 연구실, 05-3: Text Representation II - Distributed Representation Part 3 (GloVe & FastText)](https://www.youtube.com/watch?v=JZI74rrMb_M&t=470s&ab_channel=%EA%B3%A0%EB%A0%A4%EB%8C%80%ED%95%99%EA%B5%90%EC%82%B0%EC%97%85%EA%B2%BD%EC%98%81%EA%B3%B5%ED%95%99%EB%B6%80DSBA%EC%97%B0%EA%B5%AC%EC%8B%A4)

## Word2Vec의 한계
![image](https://github.com/Naver-Boostcamp-AI-Tech-6th-NLP/Study_is_all_you_need/assets/71856506/019e74e6-b1aa-4c77-9502-3fdbaae7bf6d)
- Word2Vec는 매우 빈번하게 자주 사용되는 단어(The, A 등)을 학습하는데 많은 시간을 씀
- 위의 문단에서 The가 매우 빈번하게 등장, 학습의 불균형 발생

## GloVe Notation
![image](https://github.com/Naver-Boostcamp-AI-Tech-6th-NLP/Study_is_all_you_need/assets/71856506/80abfe69-ccc6-4d38-b51d-c77be204f356)
- X는 동시발생행렬로, 총 단어수 제곱의 차원 가짐 (사이즈 엄청 큼)
- $X_{ij}$는 단어 i와 j가 동시에 발생하는 빈도
- $P_{ij}=P(j|i)$, 즉 i가 등장했을 때, j가 같이 등장할 조건부확률
- w와 context word $ \tilda{w} $를 함께 이용해서 임베딩 벡터를 찾아가는 과정

## GloVe 임베딩 예시

![image](https://github.com/Naver-Boostcamp-AI-Tech-6th-NLP/Study_is_all_you_need/assets/71856506/c0460ed3-078c-4fbf-a6fe-22ebef26cf62)

- 첫 째 열을 보면 단어 solid는 steam 보단, ice와 함께 등장할 확률이 높음
- 때문에 $\frac{P(k|ice)}{P(k|steam)}$ 가 1 보다 큼
- 반면 k가 water면, ice와 steam 모두와 비슷한 확률값 가짐

## GloVe Formulation
![image](https://github.com/Naver-Boostcamp-AI-Tech-6th-NLP/Study_is_all_you_need/assets/71856506/bb22846f-f558-4387-95e5-ee47c6728f55)
- 위의 생각을 조금만 확장하면, 두 단어와 context word k의 관계를 분수 조건부확률로 나타낼 수 있음
- context word는 두 단어와 관련성이 있는지 추정하는 단어
- GloVe를 구체화하기 위해, 세 단어의 관계를 나타내는 F 함수를 나타내야 함
- 이 때 아래 세가지 스킬 사용
  - $w_i$와 $w_j$의 차이를 이용해서 변수 3개 -> 2개
  - 내적연산을 통해 $w_i$와 $w_j$의 차이와 context word를 관계시키자!
  - Homomorphism: 입력을 덧셈의 항등원으로 바꿔주면, Output은 곱셈의 항등원으로 나오게 만들자!


## Homomorphism(준동형 사상) 
- 어려우면 스킵 추천
- Homomorphism: 입력을 덧셈의 항등원으로 바꿔주면, Output은 곱셈의 항등원으로 나오게 만들자!

![image](https://github.com/Naver-Boostcamp-AI-Tech-6th-NLP/Study_is_all_you_need/assets/71856506/f650af35-4e95-486e-8da3-11d087c7fd7f)

![image](https://github.com/Naver-Boostcamp-AI-Tech-6th-NLP/Study_is_all_you_need/assets/71856506/a2750715-5471-4cc4-833b-0900fd845bd8)

- 다시 말하면, ice-steam을 인풋으로 넣으면 steam-ice를 인풋으로 넣을 때랑 아웃풋이 역수로 나와야 함

![image](https://github.com/Naver-Boostcamp-AI-Tech-6th-NLP/Study_is_all_you_need/assets/71856506/8ceecc37-6d3b-471e-a754-de07ac0ec18b)

- 즉, 입력공간에서 덧셈의 항등원과 출력공간에서 곱셈의 항등원을 매핑시킬 수 있는 함수 F가 필요
- 위의 예시처럼 F 안에서 덧셈 연산을 F 밖의 곱셈 연산으로 뺄 수 있음
- 이를 만족하는 가장 대표적인 함수가 exp라서 이를 GloVe 임베딩에 사용하겠다!

![image](https://github.com/Naver-Boostcamp-AI-Tech-6th-NLP/Study_is_all_you_need/assets/71856506/efe8062f-0bb4-49c8-81b0-b4893627a037)

- 결국, 수학 연산을 통해 풀어보고 위처럼 $ log X_i $를 두 상수항 취급하면 아래와 같은 결론 얻음
- **우리가 얻고자 하는 두 단어 i와 k 임베딩의 내적 + 상수 = i와 k의 동시발생확률의 log**
  

## GloVe Objective Function
![image](https://github.com/Naver-Boostcamp-AI-Tech-6th-NLP/Study_is_all_you_need/assets/71856506/91413e0d-83bd-4480-817a-bb8d485611fa)

- 빨간색 부분 (동시발생확률)은 우리가 관측을 통해 얻는 값
- 임베딩 벡터 $w_i, w_j$와 상수항 $b_i, b_j$는 학습을 통해 찾아야하는 값
- 임베딩 내적과 동시발생확률의 log 차이를 최소화시키는 목적함수
- (심화) 실제 위의 방법대로 학습시 자주 등장하는 단어에 대해 과도하게 가중치 반영됨
- high-frequency word에 대해 가중치 내려주기 위해 추가 term 붙힘


![image](https://github.com/Naver-Boostcamp-AI-Tech-6th-NLP/Study_is_all_you_need/assets/71856506/ad1a99ba-a1b1-4526-a15d-1f925293e93a)

- 위에서 말한 가중치 내려주기 위한 식 및 그래프

## Result

![image](https://github.com/Naver-Boostcamp-AI-Tech-6th-NLP/Study_is_all_you_need/assets/71856506/0dab49a5-9a3e-4e50-9b2d-edffeba65d29)

- GloVe를 통해 학습 결과 어려운 개구리 학명들이 서로 가까운 거리 보임
- 이외에도 아래처럼 여러 단어의 관계가 잘 보존됨


