import sys
sys.path.append('.')
from gan import GanMnist
if __name__ == '__main__':
    mnist = GanMnist()
    mnist.train()
