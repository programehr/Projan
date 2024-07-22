import torch.nn
import torch.nn.functional as F

from ..backdoor_defense import BackdoorDefense
from trojanzoo.utils import env

'''
This is not a defense. This is used to evaluate a model.
For example it can get the raw class scores (before taking argmax)...
'''

class CheckModel(BackdoorDefense):
    name: str = 'check_model'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def detect(self, **kwargs):
        print(f'Clean Confidence: {self.detect0()}')
        if self.attack.name == 'badnet':
            print(f'Trigger Confidence: {self.detect0(stamp="badnet")}')
        elif self.attack.name == 'prob':
            for i in range(self.attack.nmarks):
                print(f'Trigger ({i+1}) Confidence: {self.detect0(index=i)}')
        elif self.attack.name == 'ntoone':
            print(f'Trigger Confidence: {self.detect0(stamp="all")}')

    def detect0(self, index=None, stamp=None):
        val_loader = self.dataset.get_dataloader(mode='test', drop_last=True)
        self.model.eval()
        all_confs = torch.empty((0,))
        all_stds = torch.empty((0,))
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                x, y = data
                x = x.to(env['device'])
                if index is not None:
                    x = self.attack.add_mark(x, index=index)
                elif stamp=='all':
                    x = self.attack.add_mark(x, stamp_mode='all', index=None)
                elif stamp == 'badnet':
                    x = self.attack.add_mark(x)
                y = y.to(env['device'])
                out = self.model(x)
                # sm_out = F.softmax(out, 1)
                sm_out = out
                conf = sm_out.max(1)[0].cpu()
                std = sm_out.std(1).cpu()
                all_confs = torch.cat((all_confs, conf), 0)
                all_stds = torch.cat((all_stds, std), 0)
        return all_confs.mean().item(), all_stds.mean().item()


