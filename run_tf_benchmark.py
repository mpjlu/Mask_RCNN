#!/usr/bin/python
from argparse import ArgumentParser
from platform_util import platform
from model_init import model_initializer

DEFAULT_INTEROP_VALUE_ = 2

class benchmark_util:

  def main(self):
    p = platform()
    arg_parser = ArgumentParser(description='Launchpad for Parameterized Docker'
        ' builds')

    arg_parser.add_argument('-b', "--batch-size",
                            	help="Specify the batch size. If this " \
				"parameter is not specified or is -1, the " \
				"largest ideal batch size for the model will " \
				"be used.",
                            	dest="batch_size", type=int, default=-1)
    arg_parser.add_argument('-n', "--num-cores",
                            	help='Specify the number of cores to use. ' \
				'If the parameter is not specified ' \
				'or is -1, all cores will be used.',
                            dest="num_cores", type=int, default=-1)
#This adds support for a --single-socket param with a default value of False. 
#Only if '--single-socket' is on the command-line will the value be true.
    arg_parser.add_argument('-s','--single-socket', 
				help='Indicates that only one socket should ' \
				'be used. If used in conjunction with ' \
				'--num-cores, all cores will be allocated ' \
				'on the single socket.',
                            	dest="single_socket", action='store_true')
#This adds support for a --inference-only param with a default value of False. 
#Only if '--inference-only' is on the command-line will the value be true.
    arg_parser.add_argument('-f', "--inference-only", 
				help='Only do inference.', 
				dest='inference_only',
				action='store_true')
    arg_parser.add_argument('-c', "--checkpoint",
                            help='Specify the location of checkpoint/training model. ' \
                                'If --forward-only is not specified, training model/weights will be ' \
                                'written to this location. If --forward-only is specified, ' \
                                'assumes that the location ' \
                                'points to a model that has already been trained. ',
                            	dest="checkpoint", default=None)
    arg_parser.add_argument("-d", "--data-location", 
				help="Specify the location of the data. " \
				"If this parameter is not specified, " \
				"the benchmark will use random/dummy data.",
                            	dest="data_location", default=None)
    arg_parser.add_argument('-a', "--num_intra_threads", type=int, 
				help="Specify the number of threads within the layer", 
				dest="num_intra_threads",
                          	default=(p.num_cores_per_socket() * p.num_cpu_sockets()))
    arg_parser.add_argument('-e', "--num_inter_threads", type=int, 
				help='Specify the number threads between layers', 
				dest="num_inter_threads",
                          	default=DEFAULT_INTEROP_VALUE_)
    arg_parser.add_argument('-v', "--verbose", 
				help='Print verbose information.', 
				dest='verbose',
				action='store_true')
    args = arg_parser.parse_args()

    mi = model_initializer(args)
    mi.run()

if __name__ == "__main__":
  util = benchmark_util()
  util.main()

