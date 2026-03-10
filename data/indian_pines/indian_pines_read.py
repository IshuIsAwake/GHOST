import scipy.io
data = scipy.io.loadmat('Indian_pines_corrected.mat')
print(data['indian_pines_corrected'].shape) 