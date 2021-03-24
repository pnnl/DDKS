# ddKS

Code repo for the d-dimensional Kolmogorov-Smirnov test.

As of 3/24/2021 there are 3 methods implemented:

* ddKS - d-dimensional KS test caclulated per [link publication] 
    * Variable splitting of space (all points, subsample, grid spacing)
* rdKS - ddKS approximation using distance from 2^d corners [link publication]
* vdKS - ddKS approximation calculating ddks distance between voxels instead of points

All xdks functions expect inputs of the form xdks(pred: [N1,d], true: [N2,d]) where N1,N2 are the number of samples and d is the dimension of the data.
    
    #Initilize instance of ddks
    xdks = ddks.methods.ddKS()
    #Generate two data sets
    pred = torch.normal(0.0,1.0,(100, 4))
    true = torch.normal(0.0,1.0,(100, 4))
    D = xdks(pred,true)


### Directories:
1. methods - Callable classes for xdks methods [x=d,r,v]
1. data - Contains several data generators to play around with
1. run_scripts - Contains an example run script 
1. Unit_tests - Contains unit tests for repo   
