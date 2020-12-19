import torch
import numpy as np
'''
Generate Toy problem dataset 
'''




class SmallDataSet:
    def __init__(self,N, Nper, dim, input_bounds, output_bounds,addvar=True,det_dim=2):
        self.N = N # Number of Different Launch points
        self.Nper = Nper # Number runs per launch point
        self.dim = dim
        self.input_bounds = input_bounds
        self.output_bounds = output_bounds
        self.det_dim = det_dim
        self.addvar = addvar
    def generate_data(self):
        if self.addvar:
            data_input = torch.rand([self.N,2*self.dim+1])
            data_input[:,2*self.dim] +=0.2 #Ensure number of bounces is always low
            Output = [torch.stack([torch.cat((data, self.calcDetection(torch.cat((data[:self.dim],data[self.dim+1:]))))) for i in range(self.Nper)]) for data in data_input]
        else:
            data_input = torch.rand([self.N, 2 * self.dim])
            data_input[:, 2 * self.dim - 1] += 0.2  # Ensure number of bounces is always low
            Output = [torch.stack([torch.cat((data, self.calcDetection(data.clone()))) for i in range(self.Nper)]) for
                      data in data_input]

        return torch.cat(Output,dim=0)
    def calcDetection(self,data):
        time = torch.zeros(1)
        bounds = torch.zeros(self.dim)
        #Setup bounds based on direction of particle
        for i in range(self.dim):
            if data[i+self.dim] < 0:
                bounds[i] = self.input_bounds[0, i]
            else:
                bounds[i] = self.input_bounds[1,i]
        while (data[self.det_dim] > self.input_bounds[0, self.det_dim] and data[self.det_dim] < self.input_bounds[1, self.det_dim]):
            # Calculate First boundary hit  (boundary-position)/velocity
            hit_t = (bounds-data[:self.dim])/data[self.dim:]
            min_t = hit_t.min()#Or torch.min
            arg_t = hit_t.argmin()
            if bounds[arg_t] == self.input_bounds[0,arg_t]:
                bounds[arg_t] = self.input_bounds[1,arg_t]
            else:
                bounds[arg_t] = self.input_bounds[0, arg_t]
            #Update Positions and velocities
            data[:self.dim] += min_t*data[self.dim:]
            time += min_t
            data[self.dim+arg_t] *= -1
            #Add variation
            if self.addvar:
                data[self.dim:] = self.addVariation(data[self.dim:])
        #Create output by removing detection dimension and adding time to output tensor
        tmp = torch.cat((data[:self.dim],time))
        output =torch.cat((tmp[:self.det_dim],tmp[self.det_dim+1:]))
        return output
    def addVariation(self,vels):
        thetas = torch.randn((3))*np.pi/90
        RX = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(thetas[0]), -torch.sin(thetas[0])],
            [0, torch.sin(thetas[0]), torch.cos(thetas[0])]
        ])
        RY = torch.tensor([
            [torch.cos(thetas[1]), 0, torch.sin(thetas[1])],
            [0, 1, 0],
            [-torch.sin(thetas[1]), 0, torch.cos(thetas[1])]
        ])
        RZ = torch.tensor([
            [torch.cos(thetas[2]), -torch.sin(thetas[2]), 0],
            [torch.sin(thetas[2]), torch.cos(thetas[2]), 0],
            [0, 0, 1]
        ])

        if vels.shape == 2:
            vels = torch.cat(vels,[0])
            return torch.matmul(vels,RZ)
        return vels.matmul(RX.matmul(RY.matmul(RZ)))

