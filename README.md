## 🌠 Stellar: An Unreal Engine-Based Large Screen & OptiTUIO Multitouch Game / Stellar : Unreal Engine 기반 대형스크린 & OptiTUIO 멀티터치 게임

**Stellar** is a cooperative multiplayer game for three or more players, characterized by easy controls and fast-paced gameplay.

* Platform: Windows  
* Engine: Unreal 5.3.2

Stellar는 3인 이상의 멀티플레이어 협동 게임으로, 쉬운 조작과 빠른 게임 진행이 특징입니다.

* 플랫폼: Windows
* 사용엔진: Unreal 5.3.2

![image](https://github.com/user-attachments/assets/5180ec91-8a99-4b28-9bba-840ad3534b87)

## Team Introduction / 팀원 소개

* [KIM HYUN MIN 김현민](https://github.com/hyunmin0809)
* [LEE JU YEON 이주연](https://github.com/Yongmalyang/)
* [JOO SO YEON 주소연](https://github.com/dubidubob)

## Table of Contents / 목차 

[Game Description / 게임 설명](https://github.com/OctoNeverDie/6mScreen-MultiTouch-Game-2025/blob/main/README.md#game-description--%EA%B2%8C%EC%9E%84-%EC%84%A4%EB%AA%85)  
[Key Features / 주요 기능](https://github.com/OctoNeverDie/6mScreen-MultiTouch-Game-2025/blob/main/README.md#key-features--%EC%A3%BC%EC%9A%94-%EA%B8%B0%EB%8A%A5) 
 
  [1. Tutorial / 튜토리얼](https://github.com/OctoNeverDie/6mScreen-MultiTouch-Game-2025/blob/main/README.md#1-tutorial--%ED%8A%9C%ED%86%A0%EB%A6%AC%EC%96%BC)  
  [2. Multitouch Touch Screen / 멀티터치 터치 스크린](https://github.com/OctoNeverDie/6mScreen-MultiTouch-Game-2025/blob/main/README.md#2-multitouch-touch-screen--%EB%A9%80%ED%8B%B0%ED%84%B0%EC%B9%98-%ED%84%B0%EC%B9%98-%EC%8A%A4%ED%81%AC%EB%A6%B0)  
  [3. Grading / 판정 시스템](https://github.com/OctoNeverDie/6mScreen-MultiTouch-Game-2025/blob/main/README.md#3-grading--%ED%8C%90%EC%A0%95-%EC%8B%9C%EC%8A%A4%ED%85%9C)  
  [4. Game Sequence / 게임 시퀀스](https://github.com/OctoNeverDie/6mScreen-MultiTouch-Game-2025/blob/main/README.md#4-game-sequence--%EA%B2%8C%EC%9E%84-%EC%8B%9C%ED%80%80%EC%8A%A4)  

[Principle of Optical Illusion / 착시 원리](https://github.com/OctoNeverDie/6mScreen-MultiTouch-Game-2025/blob/main/README.md#principle-of-optical-illusion--%EC%B0%A9%EC%8B%9C-%EC%9B%90%EB%A6%AC)  
[Result Report / 결과 보고](https://github.com/OctoNeverDie/6mScreen-MultiTouch-Game-2025/blob/main/README.md#result-report--%EA%B2%B0%EA%B3%BC-%EB%B3%B4%EA%B3%A0)  
[Potential for Development / 발전 가능성](https://github.com/OctoNeverDie/6mScreen-MultiTouch-Game-2025/blob/main/README.md#potential-for-development--%EB%B0%9C%EC%A0%84-%EA%B0%80%EB%8A%A5%EC%84%B1)  

## Game Description / 게임 설명

This game is a cooperative multiplayer game for three or more players. One player takes on the role of the conductor, while the other players perform the movements.  
The conductor stands at the back, observes the optical illusion, and describes the shape. The players in front then input movements to match the described shape.  
If they hold a specific movement for three seconds, one puzzle is cleared.  
The key challenge was ensuring that the LiDAR sensor-based multitouch system accurately captures fast inputs, allowing for a dynamic and fast-paced gameplay experience.

이 게임은 3인 이상의 멀티플레이어 협동 게임입니다. 1명이 지휘를, 나머지 사람들이 동작을 맡습니다.
지휘자 1명이 뒤에 서서 착시 이미지를 보고 형태를 설명하면, 앞에 선 나머지 사람들이 형태에 맞게 input을 넣습니다.
특정 동작에서 정지하고 3초간 홀드하면 한 개의 퍼즐이 클리어됩니다.
라이다 센서 기반 멀티터치를 잘 받아와서 속도감 있는 게임으로 제작하는 것이 관건이었습니다.

## Key Features / 주요 기능

### 1. Tutorial / 튜토리얼

![image](https://github.com/user-attachments/assets/23ec2331-6719-42e1-b304-033ada46b206)

Before the game starts, a short tutorial video is played.  
It introduces role distribution among players, game rules, and screen touch techniques.

게임 시작 전, 짧은 게임 방법 소개 영상이 재생됩니다. 
함께 플레이할 유저들과의 역할 분담, 게임의 규칙, 스크린 터치 요령 등을 안내합니다. 

### 2. Multitouch Touch Screen / 멀티터치 터치 스크린

![image](https://github.com/user-attachments/assets/f72efbbe-2390-4703-a899-35518854b46c)

* The screen displays an optical illusion image containing hidden shapes.  
  * (Refer to the "Optical Illusion Principle" section for details.)  
* Players standing close to the screen may find it difficult to spot the shapes, while the player standing at the back can see approximately 2 to 5 points connected by lines.  
* The player at the back communicates the positions to the players in front.  
  * Example: "Raise your left hand a bit higher," "Move your right foot slightly to the left," etc.  
* After correctly positioning, the players must maintain the pose for 3 seconds.  
  * At this time, the number of touches within a specified range must match the number of points to trigger the validation.

* 화면 상에는 특정 그림이 숨겨진 착시 이미지가 배치됩니다.
  * (착시에 관한 내용은 '착시 원리' 부분 참고) 
* 가까이 서 있는 사람은 찾기 쉽지 않지만, 뒤에 서 있는 사람은 대략 2 ~ 5개의 점과 그들을 잇는 선이 보이게 됩니다.
* 뒷사람이 앞사람에게 위치를 전달합니다.
  * 예: 네 왼손을 좀 더 위로 올려, 오른발을 조금만 왼쪽으로 이동해 봐, 등등
* 제대로 입력을 한 후, 그 상태를 3초간 유지해야 합니다.
  * 이때, 점의 개수와 동일한 개수의 터치 수가 일정 범위 내에 입력되어야만 판정을 시작합니다. 

### 3. Grading / 판정 시스템

* Once the validation conditions are met, a 3-second countdown begins.  
* The accuracy is measured by how close the user's input is to the star pattern.  
* The accuracy takes into account the number of inputs and the distance between each input and the stars.  
* If the number of inputs changes or the inputs move further from the points during the 3-second countdown, the process will restart from the beginning.  
* If the 3-second countdown is completed successfully, the original form of the optical illusion will fade in, and the accuracy of the puzzle will be evaluated.  
* Based on the accuracy, players can earn the labels: PERFECT, GOOD, or BAD.

* 판정 조건이 달성되면 3초간 카운트다운이 시작됩니다. 
* 정확도 측정 공식으로 유저의 input이 얼마나 별자리와 가까웠는지를 측정합니다.
* 정확도에는 input의 개수와 각 input과 별 사이의 거리가 반영됩니다.
* 3초의 카운트다운을 버티지 못하고 input의 개수가 점 개수와 달라지거나 멀어지면, 처음부터 다시 시작입니다.
* 3초의 카운트다운을 버티면 착시 이미지의 원래 모습이 페이드인으로 나타나며, 해당 퍼즐에 대한 정확도를 평가해줍니다.
* 정확도에 따라 PERFECT, GOOD, BAD 문구를 얻을 수 있습니다.

### 4. Game Sequence / 게임 시퀀스

* Steps 2 and 3 are repeated throughout the game.  
* After clearing all 15 stages or when the time limit of 500 seconds has elapsed, the final score screen will appear.

* 위의 2, 3번을 반복하여 플레이합니다.
* 15개의 스테이지를 모두 클리어하거나, 제한시간인 500초가 다 지나면 최종 스코어 화면이 나타납니다. 

## Principle of Optical Illusion / 착시 원리

[깃허브 링크](https://github.com/dubidubob/IllusionTest)

1. 배경화면 생성

    > 전체 화면에 고주파 패턴이 있어야 한다.
    > 고주파는, 밝기 채도 대비 전부 낮아야 한다. (앞사람 눈 고통 이슈)
    
    * **무작위 컬러 노이즈 이미지 생성** (3840px x 2160px, 평균 128 회색, 분산 15)
    * **전체 채도 낮춤** (BGR → HSV → BGR, 원래 채도의 20%)
    * **전체 대비 낮춤** (중간값 128과의 거리를 0.5만큼 줄인다)
       
    <img src="https://github.com/user-attachments/assets/491e416a-c4d1-45d7-9925-743ae55ddc1e" width="300">

2. 별자리 상 생성
    
    Clip Studio로 수작업
    
    * 3480px x 2160px
    * 엣지로만 작업, 꼭짓점 잘 표현됐는지 확인 
    * 이후 전체에 Gaussian Blur 100 적용
       
    <img src="https://github.com/user-attachments/assets/67f6367c-8d8b-44dd-a9f1-0aada081c3d4" width="300">

3. 배경화면에 별자리 상 합침
    
    > 배경은 고주파로, 별자리는 저주파로 만든다.
    > 배경에 묻힌 저주파로 만들기 위해 밝기, 페더링, 알파값 조절
    
    * 배경화면, 별자리 상 불러오기
    * **상의 색을 배경의 평균색으로** 전부 바꾸기 (알파값 채널은 적용하지 않음, 고유의 알파값을 유지시킨다)
    * **상의 밝기 더 어둡게 만들기**(dark-20)
    * 상에 **가우시안 블러**를 이용해 그 테두리를 희미하게 만든다. (**페더링**)
    (테두리로 갈 수록 **alpha값 0**에 수렴 + **rgb는 해당 배경의 색평균값**에 수렴)
    * **상의 알파값 높이기**(투명하게 만든다, *0.2)
    * 이후 배경 위에 처리된 상을 합성
       
    <img src="https://github.com/user-attachments/assets/3cdfa1c7-53d2-4d1e-929b-7801f268739c" width="300">

4. 최종 이미지 지표값 도출

    > **배경화면과 별자리 상의 대비 차이**
    > : 멀리서 봤을 때 얼마나 잘 보이는지 지표
    > **75% 퍼센트 확대했을 때 색 분포**
    > : 가까이서 봤을 때 얼마나 잘 안 보이는지 지표
    >    * 색 분포가 제일 안 고른 값(색 분포 표준편차 최댓값)
    >    * 색 분포의 평균(색 분포 표준편자 평균값)
    >    * 색 분포 검사 횟수
    
   첫째, 배경화면과 상의 대비
    
    배경화면은 BGR로 저장,
    별자리 상은 BGRA로 저장된다.
    
    즉, Alpha가 0에 가까우냐 아니느냐로 배경화면과 상 분리 가능
    
    * Gray Scale로 변환
    * 알파값 기반 상 마스크, 배경 마스크 분리(alpha > 0.05시 상)
    * 각 영역 평균 밝기 계산
    * 대비 평균값 차이(절대값) 도출

   둘째, 색 분포 
    
    대부분의 상이 가운데에 맺혀으니 가운데만 검사한다.
    세로 가운데 80%, 가로 가운데 80% 범위에 한정하여 도출한다.
   
    예시 플로우 : 
    * 가로 : 200, 세로 : 100일 때, 75퍼센트 확대해서 보면, 가로 : 50, 세로 : 25
    * 세로의 양쪽 40% 버리고 세로 20 ~ 80 한정
    * 가로의 양쪽 40% 버리고 가로 40 ~ 160 한정
    * 세로 20 ~ 45 가로 40~90에 위치하는 이미지의 색 분포를 검사
    * 세로 20과 45의 중간값인 32.5 ~ 57.5 가로는 40 ~ 90의 중간값인 65 ~ 115 위치하는 이미지의 색분포를 검사
    * 범위의 끝에 다다랐을 때는 끝 위치를 기준으로 한 번 더 검사 
    * 세로 55 ~ 80 가로 110 ~ 160에서 검사
    * 이후 이 중 가장 색 분포가 고르지 않은 값을 도출
    * 몇 번 색 분포도 검사했는지 횟수 도출
    * 이때까지 검사한 색 분포도의 평균을 도출

       
    <img src="https://github.com/user-attachments/assets/eca0269e-b8a3-4e32-b5c5-7bf672cac72c" width="300">



## Result Report / 결과 보고
[![시연 동영상](http://img.youtube.com/vi/chSA9CkVb6g/0.jpg)](https://youtu.be/chSA9CkVb6g)


## Potential for Development / 발전 가능성

* 대형 스크린을 활용한 설치 전시 가능
  * 예: 박물관 - 유물, 기업 - 제품 등으로 착시 이미지 대체
* 착시 이미지의 원리 및 공식 연구 가능
