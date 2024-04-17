import os, sys
sys.path.append('/home/minghao/Research/Projects/JWST/qsohost_imaging/code/image_fitting/')

from run_fitting import *

#do_one_quasar_step_one('J0148+0600', filters=['F356W'], visits=[1,2,3,4], mc_args={'burn': 20000, 'iterations': 1000, 'chains': 60})
#do_one_quasar_step_one('J0148+0600', filters=['F200W', 'F115W'], visits=[1,2,3,4], mc_args={'burn': 20000, 'iterations': 1000, 'chains': 60})
#do_one_quasar_step_two('J0148+0600', bestband='F356W', otherbands=['F200W', 'F115W'], visits=[1, 2, 3, 4], mc_args={'burn': 20000, 'iterations': 1000, 'chains': 50}) 

# first generate the cutout images

# then run fitting

# then save the results

# do the 1-PSF fitting

