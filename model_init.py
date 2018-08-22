import os
from argparse import ArgumentParser
from platform_util import platform

os.environ["KMP_BLOCKTIME"] = "1"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
os.environ["KMP_HW_SUBSET"] = "1T"
#os.environ["OMP_PROC_BIND"]="true"
# DEFAULT_INTEROP_VALUE_ = 2
class model_initializer:
  '''Add code here to detect the environment and set necessary variables before launching the model'''
  args=None
  def getCpu(self):
    p = platform()
    cores = p.num_cores_per_socket()
    if self.args.num_cores != -1:
      cores = self.args.num_cores

    self.additional_args.num_intra_threads = cores
    os.environ["OMP_NUM_THREADS"] = str(cores)
    self.additional_args.num_inter_threads = 1 
    
    cpu = '0'
    if p.num_numa_nodes() == 2 and self.args.socket_id == 1:
      print(str(cpu))
      if cores > 1:
        cpu = p.num_cores_per_socket() + '-' + str(p.num_cores_per_socket() + cores -1)
    else:
      if cores > 1:
        cpu = '0-' + str(cores-1)
    return cpu

  def __init__(self, args, custom_args=[]):
    self.args = args
    self.custom_args = custom_args
    self.command_prefix = ''

    p = platform()

    # print(self.args.cores)

    arg_parser = ArgumentParser(description='Parse additional custom args')
    arg_parser.add_argument('-a', "--num_intra_threads", type=int,
                        help="Specify the number of threads within the layer",
                        dest="num_intra_threads",
                        default=(p.num_cores_per_socket() * p.num_cpu_sockets()))
    arg_parser.add_argument('-e', "--num_inter_threads", type=int,
                        help='Specify the number threads between layers',
                        dest="num_inter_threads",
                        default=p.num_cpu_sockets())
    arg_parser.add_argument('-ac', "--accuracy_check ",
                        help='Do inference accuracy check.',
                        dest='accuracy_check',
                        action='store_true')
    arg_parser.add_argument('-gp', "--gpu_path",
                        help='Specify the location of gpu results.',
                        dest="gpu_path", default=None)
    self.additional_args = arg_parser.parse_args(self.custom_args)

    if not self.args.inference_only:
      if not self.args.single_socket:
        self.additional_args.num_inter_threads = 1
        os.environ["OMP_NUM_THREADS"] = str(p.num_cores_per_socket() * p.num_cpu_sockets())

    if self.args.verbose: 
      print('Received these args: {}'.format(self.args))
      print('Initialize here.')
    #do inference
    if self.args.inference_only:
      self.command_prefix = ' python3 coco.py evaluate ' 
      #Only for SKX 
      if self.args.single_socket:
        cpu = self.getCpu()
         
        self.command_prefix = 'numactl --cpunodebind=' + str(self.args.socket_id) + ' --membind=' + str(self.args.socket_id) + self.command_prefix
        self.command_prefix = self.command_prefix + \
                              ' --dataset=' + str(self.args.data_location) + \
                              ' --num_inter_threads ' + str(self.additional_args.num_inter_threads) + \
                              ' --num_intra_threads ' + str(self.additional_args.num_intra_threads) + \
                              ' --nw 5 --nb 50 --model=coco'
      else:
        self.command_prefix = self.command_prefix + \
                              ' --dataset=' + str(self.args.data_location) + \
                              ' --num_inter_threads ' + str(self.additional_args.num_inter_threads) + \
                              ' --num_intra_threads ' + str(self.additional_args.num_intra_threads) + \
                              ' --nw 5 --nb 50 --model=coco'
      self.command_prefix = self.command_prefix + ' --infbs ' + str(self.args.batch_size)

      if self.additional_args.gpu_path is not None:
        self.command_prefix = self.command_prefix + ' > maskrcnn_gpu.log'
      elif self.additional_args.accuracy_check:
        self.command_prefix = self.command_prefix + ' > maskrcnn_cpu.log'
    #do training
    else:
      self.command_prefix = ' python3 coco.py train '
      if self.args.batch_size == -1:
        self.args.batch_size = 2
      if self.args.single_socket:
        cpu = self.getCpu()

        self.command_prefix = 'numactl --cpunodebind=' + str(self.args.socket_id) + ' --membind=' + str(self.args.socket_id) + self.command_prefix
        self.command_prefix = self.command_prefix + \
                              ' --dataset=' + str(self.args.data_location) + \
                              ' --num_inter_threads ' + str(self.additional_args.num_inter_threads) + \
                              ' --num_intra_threads ' + str(self.additional_args.num_intra_threads) + \
                              ' --nw 5 --nb 50 --model=coco'
      else:
        self.command_prefix = self.command_prefix + \
                              ' --cp=' + self.args.checkpoint + \
                              ' --dataset=' + str(self.args.data_location) + \
                              ' --num_inter_threads ' + str(self.additional_args.num_inter_threads) + \
                              ' --num_intra_threads ' + str(self.additional_args.num_intra_threads) + \
                              ' --nw 5 --nb 50 --model=coco'
      self.command_prefix = self.command_prefix + ' --trainbs ' + str(self.args.batch_size)

  def run(self):
    print(self.command_prefix) 
    if self.command_prefix:
      if self.args.verbose:
        print("Run model here.", self.command_prefix)
      os.system(self.command_prefix)
    if self.additional_args.accuracy_check:
      print("Start accuracy check.")
      os.system('sh ./accuracy_check.sh')
