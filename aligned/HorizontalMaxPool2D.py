import torch.nn as nn

class HorizontalMaxPool2d(nn.Module):
    def __init__(self):
        super(HorizontalMaxPool2d, self).__init__()


    # params@x:输入x实际是经特征提取得到的特征图feature map
    def forward(self, x):
        inp_size = x.size()
        # 水平特征的长度为特征图的宽winp_size[3],其高度为1
        # 输出y y = nn.functional.max_pool2d(input=x,kernel_size= (1, inp_size[3]))
        return nn.functional.max_pool2d(input=x,kernel_size= (1, inp_size[3]))


# 验证测试
if __name__ == '__main__':
    import torch
    # 补充：torch.tensor()是将numpy的数组转换为tensor，而torch.Tensor()是新建一个tensor
    x = torch.Tensor(32,2048,8,4)
    hp = HorizontalMaxPool2d()
    y = hp(x)
    print(y.shape)# 得到输出为[32, 2048, 8, 1]