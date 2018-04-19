
class model_initializer:
  '''Add code here to detect the environment and set necessary variables before launching the model'''
  args=None
  def __init__(self, args):
    self.args = args
    if self.args.verbose: 
      print 'Received these args: {}'.format(self.args)
      print 'Initialize here.'

  def run(self):
    if self.args.verbose: print "Run model here."
