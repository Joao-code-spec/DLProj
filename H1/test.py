import torch

print(torch.version.cuda)
print(torch.cuda.is_available())
my_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("my_device:", my_device)

x = torch.eye(3)  # data is on the cpu 
print("By default device tensor is stored on:", x.device)

torch.cuda.get_device_name(my_device)
# you can move data to the GPU by doing .to(device)
x = x.to(my_device)  # data is moved to my_device
print("\nDevice tensor is now stored on:", x.device)  # it will still be cpu if you don't have gpu
