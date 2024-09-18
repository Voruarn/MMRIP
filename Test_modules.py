import torch
from network.SimSOD import SimSOD 

if __name__ == "__main__":
    print('Hello')

    rgb = torch.randn([1, 3, 256, 256]).cuda()
    depth = torch.randn([1, 1, 256, 256]).cuda()
    
    model = SimSOD().cuda()
    model.eval()
    output=model(rgb, depth)

    for x in output:
        print('x.shape:',x.shape)











