import sys
sys.path.append('.')
from avatar_model import AvatarModel
if __name__ == '__main__':
    avatar = AvatarModel()
    avatar.train()
     