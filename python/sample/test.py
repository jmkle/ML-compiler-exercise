import torch

torch.manual_seed(41)

class MyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.param = torch.nn.Parameter(torch.tensor([[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.]]))
        self.linear = torch.nn.Linear(4, 5)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[0., 0., 0., 0.], [5., 6., 7., 8.], [9., 10., 11., 12.], [9., 10., 11., 12.], [9., 10., 11., 12.]]))
            self.linear.bias.copy_(torch.tensor([0., 0., 0., 0., 1.]))

    def forward(self, x):
        return self.linear(x + self.param).clamp(min=0.0, max=1.0)
    
if __name__ == "__main__":
    model = MyModule()
    x = torch.tensor([[1.0, 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]])
    print(model(x))