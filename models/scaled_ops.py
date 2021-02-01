import torch
import torch.nn as nn
import torch.nn.functional as F


class Down_Joint2Part(nn.Module):
    def __init__(self):
        super(Down_Joint2Part, self).__init__()
        self.torso = [0,7,8]
        self.left_leg_up = [4]
        self.left_leg_down = [5,6]
        self.right_leg_up = [1]
        self.right_leg_down = [2,3]
        self.head = [9]
        self.left_arm_up = [10]
        self.left_arm_down = [11,12]
        self.right_arm_up = [13]
        self.right_arm_down = [14,15]
        
    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B,D,V)
        x_torso = F.avg_pool2d(x[:, :, self.torso], kernel_size=(1, 3))                                              
        x_leftlegup = x[:, :, self.left_leg_up]                              
        x_leftlegdown = F.avg_pool2d(x[:, :, self.left_leg_down], kernel_size=(1, 2))                     
        x_rightlegup = x[:, :, self.right_leg_up]                       
        x_rightlegdown = F.avg_pool2d(x[:, :, self.right_leg_down], kernel_size=(1, 2))                   
        x_head = x[:, :, self.head]                                       
        x_leftarmup = x[:, :, self.left_arm_up]
        x_leftarmdown = F.avg_pool2d(x[:, :, self.left_arm_down], kernel_size=(1, 2))                 
        x_rightarmup = x[:, :, self.right_arm_up]                    
        x_rightarmdown = F.avg_pool2d(x[:, :, self.right_arm_down], kernel_size=(1, 2))           
        x_part = torch.cat((x_leftlegup, x_leftlegdown, x_rightlegup, x_rightlegdown, x_torso, x_head,  x_leftarmup, x_leftarmdown, x_rightarmup, x_rightarmdown), dim=-1)       
        x_part = x_part.permute(0, 2, 1)  # (B,V,D)
        return x_part


class Down_Part2Body(nn.Module):

    def __init__(self):
        super(Down_Part2Body, self).__init__()
        self.torso = [4,5]
        self.left_leg = [0,1]
        self.right_leg = [2,3]
        self.left_arm = [6,7]
        self.right_arm = [8,9]
        
    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B,D,V)
        x_torso = F.avg_pool2d(x[:, :, self.torso], kernel_size=(1, 2))                                           
        x_leftleg = F.avg_pool2d(x[:, :, self.left_leg], kernel_size=(1, 2))                             
        x_rightleg = F.avg_pool2d(x[:, :, self.right_leg], kernel_size=(1, 2))                     
        x_leftarm = F.avg_pool2d(x[:, :, self.left_arm], kernel_size=(1, 2))                         
        x_rightarm = F.avg_pool2d(x[:, :, self.right_arm], kernel_size=(1, 2))                     
        x_body = torch.cat((x_leftleg, x_rightleg, x_torso, x_leftarm, x_rightarm), dim=-1)        
        x_body = x_body.permute(0, 2, 1)  # (B,V,D)      
        return x_body



class Up_Part2Joint(nn.Module):

    def __init__(self):
        super(Up_Part2Joint, self).__init__()

        self.torso = [0,7,8]
        self.left_leg_up = [4]
        self.left_leg_down = [5,6]
        self.right_leg_up = [1]
        self.right_leg_down = [2,3]
        self.head = [9]
        self.left_arm_up = [10]
        self.left_arm_down = [11,12]
        self.right_arm_up = [13]
        self.right_arm_down = [14,15]
        

    def forward(self, part):
        part = part.permute(0, 2, 1)
        N, d, w = part.size()  # [64, 128, 10]
        x = part.new_zeros((N, d, 16))

        x[:,:,self.left_leg_up] = part[:,:,0].unsqueeze(-1)
        x[:,:,self.left_leg_down] = torch.cat((part[:,:,1].unsqueeze(-1), part[:,:,1].unsqueeze(-1)),-1)
        x[:,:,self.right_leg_up] = part[:,:,2].unsqueeze(-1)
        x[:,:,self.right_leg_down] = torch.cat((part[:,:,3].unsqueeze(-1), part[:,:,3].unsqueeze(-1)),-1)
        x[:,:,self.torso] = torch.cat((part[:,:,4].unsqueeze(-1), part[:,:,4].unsqueeze(-1), part[:,:,4].unsqueeze(-1)),-1)
        x[:,:,self.head] = part[:,:,5].unsqueeze(-1)
        x[:,:,self.left_arm_up] = part[:,:,6].unsqueeze(-1)
        x[:,:,self.left_arm_down] = torch.cat((part[:,:,7].unsqueeze(-1),part[:,:,7].unsqueeze(-1)),-1)
        x[:,:,self.right_arm_up] = part[:,:,8].unsqueeze(-1)
        x[:,:,self.right_arm_down] = torch.cat((part[:,:,9].unsqueeze(-1),part[:,:,9].unsqueeze(-1)),-1)
        x = x.permute(0, 2, 1)
        return x


class Up_Body2Part(nn.Module):

    def __init__(self):
        super(Up_Body2Part, self).__init__()

        self.torso = [4,5]
        self.left_leg = [0,1]
        self.right_leg = [2,3]
        self.left_arm = [6,7]
        self.right_arm = [8,9]

    def forward(self, body):
        body = body.permute(0,2,1)
        N, d, w = body.size()  # [64, 128, 5]
        x = body.new_zeros((N, d, 10))

        x[:,:,self.left_leg] = torch.cat((body[:,:,0:1], body[:,:,0:1]),-1)
        x[:,:,self.right_leg] = torch.cat((body[:,:,1:2], body[:,:,1:2]),-1)
        x[:,:,self.torso] = torch.cat((body[:,:,2:3], body[:,:,2:3]),-1)
        x[:,:,self.left_arm] = torch.cat((body[:,:,3:4], body[:,:,3:4]),-1)
        x[:,:,self.right_arm] = torch.cat((body[:,:,4:5], body[:,:,4:5]),-1)
        x = x.permute(0, 2, 1)
        return x