import os, sys
sys.path.append('/home/minghao/Research/Projects/JWST/qsohost_imaging/code/image_fitting/')

from run_fitting import *

#do_one_quasar_step_one('J159-02', filters=['F356W', 'F200W', 'F115W'], visits=[1, 2], mc_args={'burn': 40000, 'iterations': 1000, 'chains': 50})
do_one_quasar_step_two('J159-02', bestband='F356W', otherbands=['F200W', 'F115W'], visits=[1, 2], mc_args={'burn': 40000, 'iterations': 1000, 'chains': 50}) 

# first generate the cutout images

# then run fitting

# then save the results

