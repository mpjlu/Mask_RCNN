#!/usr/bin/python
from argparse import ArgumentParser
from platform_util import platform
from model_init import model_initializer

DEFAULT_INTEROP_VALUE_ = 2

# to run this script for a model:
# python run_tf_benchmark.py -s -b=128 -m='inception_resnet_v2' -d='/dataset/mandy_dataset/TF_Imagenet_FullData' -c='/dataset/chkpt'


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
        # This adds support for a --single-socket param with a default value of False.
        # Only if '--single-socket' is on the command-line will the value be true.
        arg_parser.add_argument('-s', '--single-socket',
                                help='Indicates that only one socket should ' \
                                     'be used. If used in conjunction with ' \
                                     '--num-cores, all cores will be allocated ' \
                                     'on the single socket. If --socket-id is specified, '\
                                     'the specific socket will be used.',
                                dest="single_socket", action='store_true')
        arg_parser.add_argument('-i', "--socket-id",
                                help='Specify which socket to use. Default is socket 0.',
                                dest="socket_id", type=int, default=0)
        arg_parser.add_argument('-r', "--cores",
                                help='Specify which socket to use. Default is socket 0.',
                                dest="cores", default='0-28')
        # This adds support for a --inference-only param with a default value of False.
        # Only if '--inference-only' is on the command-line will the value be true.

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
        arg_parser.add_argument('-m', "--model-name",
                                help='Specify the model name to run benchmark for',
                                dest='model_name')
        arg_parser.add_argument('-v', "--verbose",
                                help='Print verbose information.',
                                dest='verbose',
                                action='store_true')
        arg_parser.add_argument('-rg', "--run_gpu",
                                help='Indicate whether gpu path.',
                                dest='run_gpu',
                                action='store_true')
        args, unknown = arg_parser.parse_known_args()
        mi = model_initializer(args, unknown)
        mi.run()


if __name__ == "__main__":
    util = benchmark_util()
    util.main()
