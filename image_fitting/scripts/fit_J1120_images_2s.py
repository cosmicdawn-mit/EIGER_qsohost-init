import os, sys
sys.path.append('/home/minghao/Research/Projects/JWST/qsohost_imaging/code/image_fitting/')

from run_fitting import *

#do_one_quasar_step_one('J1120+0641', filters=['F356W'], visits=[3], mc_args={'burn': 30000, 'iterations': 1000, 'chains': 40})
#do_one_quasar_step_one('J1120+0641', filters=['F200W'], visits=[1,2,3,4], mc_args={'burn': 20000, 'iterations': 1000, 'chains': 40})

#do_one_quasar_step_two('J1120+0641', bestband='F356W', otherbands=['F200W'], bestvisits=[1,2,3], visits=[1, 2, 3, 4],\
#                       mc_args={'burn': 20000, 'iterations': 1000, 'chains': 40}) 

#do_one_quasar_psf_only('J1120+0641', filters=['F356W'], visits=[1,3,4], mc_args={'burn': 2000, 'iterations': 500, 'chains': 20})


# first generate the cutout images

# then run fitting

# then save the results

