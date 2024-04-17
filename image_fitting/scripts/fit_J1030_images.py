import os, sys
sys.path.append('/home/minghao/Research/Projects/JWST/qsohost_imaging/code/image_fitting/')

from run_fitting import *

#do_one_quasar_step_one('J1030+0524', filters=['F356W', 'F200W', 'F115W'], visits=[1,3,4], mc_args={'burn': 20000, 'iterations': 1000, 'chains': 60})
do_one_quasar_psf_only('J1030+0524', filters=['F356W', 'F200W', 'F115W'], visits=[1,3,4], mc_args={'burn': 10000, 'iterations': 1000, 'chains': 20})

