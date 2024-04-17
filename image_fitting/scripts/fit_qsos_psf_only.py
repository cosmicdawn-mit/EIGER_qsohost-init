import os, sys
sys.path.append('/home/minghao/Research/Projects/JWST/qsohost_imaging/code/image_fitting/')

from run_fitting import *

#do_one_quasar_psf_only('J0148+0600', filters=['F356W', 'F200W', 'F115W'], visits=[1,2,3,4], mc_args={'burn': 10000, 'iterations': 1000, 'chains': 20})
#do_one_quasar_psf_only('J159-02', filters=['F356W', 'F200W', 'F115W'], visits=[1,2], mc_args={'burn': 10000, 'iterations': 1000, 'chains': 20})
#do_one_quasar_psf_only('J1120+0641', filters=['F356W', 'F200W', 'F115W'], visits=[1,2,3,4], mc_args={'burn': 10000, 'iterations': 1000, 'chains': 20})
do_one_quasar_psf_only('J1120+0641', filters=['F115W'], visits=[1,2,3,4], mc_args={'burn': 10000, 'iterations': 1000, 'chains': 20})


