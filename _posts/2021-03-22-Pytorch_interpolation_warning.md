---
layout: post
title: Pytorch interpolation warning
hero_image: https://github.com/SeongjaeHong/SeongjaeHong.github.io/blob/master/img/leaf.jpeg?raw=true
hero_height: is-large
hero_darken: true
image: https://github.com/SeongjaeHong/SeongjaeHong.github.io/blob/master/img/interpolation.png?raw=true
tags: pytorch
comments: true
---
Pytorch에서 보간법을 사용하기 위해서 [torch.nn.functional.interpolation](https://pytorch.org/docs/stable/nn.functional.html#interpolate) 을 사용한다.

interpolation 입력인자 중에 align_corners 라는게 있는데, pytorch 0.3.1 이후 버전부터 `align_corners`의 기본 값이 `True`에서 `False`로 변경되었다.

pytorch 0.3.1 보다 더 최신 버전에서 interpolation을 사용할 경우 경고 메세지가 뜨면서 `align_corners`의 기본 값이 바뀌었다는 것을 알려준다.
```
UserWarning: Default upsampling behavior when mode=bilinear is changed to align_corners=False since 0.4.0.
Please specify align_corners=True if the old behavior is desired.
See the documentation of nn.Upsample for details.
```
경고 메세지를 보고 싶지 않은 경우 `align_corners=True`로 변경해주면 더 이상 경고 메세지가 뜨지 않는다.

하지만 무턱대고 값을 변경하기보다는 `align_corners`가 도대체 무엇인지 한 번 알아보기로 했다.

Pytorch docs에서는 다음과 같이 설명한다.
```
Align_corners=True 인 경우, 각 코너 픽셀의 중심점을 기준으로 텐서를 정렬하고, 코너 픽셀의 값을 보존한다.
반면에 align_corners=False 인 경우, 각 코너 픽셀을 기점으로 텐서를 정렬하고,
코너 픽셀의 값을 이용해 패딩 영역의 픽셀을 생성한다.
```
글만 봐서는 무슨 소리인지 한 번에 알기가 어렵다.

구글 검색을 해본 결과 다음 글을 찾을 수 있었다. ([링크](https://discuss.pytorch.org/t/what-we-should-use-align-corners-false/22663/9))

<center>
<img src="https://github.com/SeongjaeHong/SeongjaeHong.github.io/blob/master/img/interpolation.png?raw=true">
</center>

`align_corners=True`인 경우, 보간법으로 새로 생성된 픽셀을 정렬할 때, 픽셀 간격을 양 모서리 끝 픽셀을 기준으로 정한다.

따라서 각 픽셀 간격이 동일하게 떨어진다.

반면에 `algin_corners=False`인 경우, 기존 픽셀을 중심으로 새롭게 생성된 픽셀들이 배치된다. 

따라서 각 픽셀 간격이 동일하지가 않다.


실제 테스트를 해보면 다음과 같은 결과를 확인할 수 있다.

```
import torch
import torch.nn.functional as F
import numpy as np

x = torch.from_numpy(np.array([[5,2,3],[4,10,6],[7,8,9]], dtype=np.float32)[None, :])
align_true = F.interpolate(t, scale_factor=2, mode='linear', align_corners=True)
align_false = F.interpolate(t, scale_factor=2, mode='linear', align_corners=False)

print(align_true)
print(align_false)
```
```
tensor([[[5.0000, 3.8000, 2.6000, 2.2000, 2.6000, 3.0000],
         [4.0000, 6.4000, 8.8000, 9.2000, 7.6000, 6.0000],
         [7.0000, 7.4000, 7.8000, 8.2000, 8.6000, 9.0000]]])

tensor([[[5.0000, 4.2500, 2.7500, 2.2500, 2.7500, 3.0000],
         [4.0000, 5.5000, 8.5000, 9.0000, 7.0000, 6.0000],
         [7.0000, 7.2500, 7.7500, 8.2500, 8.7500, 9.0000]]])
```

`align_ture`의 픽셀 간격은 각 행마다 동일하게 유지되는 반면, 

`align_false`는 간격이 동일하지 않은 것을 알 수 있다.

그렇다면 실제 모델 학습 시에는 `align_corners=True`를 이용하는 것이 더 좋을까?

검색 결과, Segmentation 모델을 학습할 경우 `align_corners=True`일 때 더 좋은 결과가 나온다는 의견이 자주 보였다.

하지만 [해당 글](https://github.com/pytorch/vision/issues/1708#issuecomment-620049255) 에서 간단한 실험을 해본 결과, 유의미한 성능 차이가 있지는 않다고 한다.

따라서 경고 메세지를 보고 싶지 않은 사람은 그냥 `align_corners=True`을 해도 상관 없을 것으로 보인다.



 
