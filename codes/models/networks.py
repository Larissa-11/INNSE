import logging
import math

from models.modules.IVnv_arch import *
from models.modules.Subnet_constructor import subnet

logger = logging.getLogger('base')


####################
# define network
####################
def define_G(opt):
	model = opt['model']
	opt_net = opt['network_G']
	which_model = opt_net['which_model_G']
	subnet_type = which_model['subnet_type']
	# opt_datasets = opt['datasets']

	if opt_net['init']:
		init = opt_net['init']
	else:
		init = 'xavier'

	down_num = int(math.log(opt_net['scale'], 2))

	if model == 'V_IRN':
		netG = Net(opt, subnet(subnet_type, init), down_num)
		return netG
	else:
		netG = InvRescaleNet(opt_net['in_nc'], opt_net['out_nc'], subnet(subnet_type, init), opt_net['block_num'],
							 down_num)

		# netG = InvNN(opt_net['in_nc'], opt_net['out_nc'], subnet(subnet_type, init), opt_net['block_num'],
		# 					 down_num)
		return netG
