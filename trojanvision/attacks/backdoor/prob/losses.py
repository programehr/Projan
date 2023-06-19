import torch
from torch import tensor
import torch.nn.functional as F
from trojanzoo.environ import env


def smooth(t, eps=1e-3):
    output = torch.clone(t)
    if t.ndim == 2:
        bsize, n = t.shape

        for i in range(bsize):
            for j in range(n):
                if t[i, j] < eps:
                    output[i, j] = eps
    elif t.ndim == 1:
        n = t.shape[0]
        for j in range(n):
            if t[j] < eps:
                output[j] = eps
    else:
        raise ValueError()
    return output


def CE_btw_distros(a, b):
    p = smooth(a)
    q = smooth(b)
    log_p = torch.log(p)
    log_q = torch.log(q)
    return (- p * log_q - q * log_p).mean()


def sum_loss(l1, l2):
    lsum = lambda output, mod_outputs, label, target, probs: \
        l1(output, mod_outputs, label, target, probs) + \
        l2(output, mod_outputs, label, target, probs)
    return lsum


def coeff_loss(coeff, l):
    lcoeff = lambda output, mod_outputs, label, target, probs: \
        coeff * l(output, mod_outputs, label, target, probs)
    return lcoeff


# Cross Entropy between output and label
def loss1(output, mod_outputs, label, target, probs):
    return torch.nn.CrossEntropyLoss()(output, label)


# Cross Entropy between output and target
def loss2_0(output, mod_outputs, label, target, probs):
    CE1 = torch.nn.CrossEntropyLoss()(output, label)
    target_vec = target * torch.ones_like(label)
    CE2 = torch.nn.CrossEntropyLoss()(output, target_vec)
    return CE1


# average of loss1 and loss2_0
def loss2(output, mod_outputs, label, target, probs):
    CE1 = torch.nn.CrossEntropyLoss()(output, label)
    target_vec = target * torch.ones_like(label)
    CE2 = torch.nn.CrossEntropyLoss()(output, target_vec)
    return 0.5 * (CE1 + CE2)


# quadratic poison loss
def loss2_1(output, mod_outputs, label, target, probs):
    n = len(mod_outputs)
    smouts = [None] * n
    part_loss = [None] * n
    for i in range(n):
        smouts[i] = F.softmax(mod_outputs[i], 1)
        part_loss[i] = (smouts[i][:, target].sum() - smouts[i].shape[0] * probs[i]) ** 2
        '''
        target_tensor = target*torch.ones_like(label, device=env['device'])
        part_loss[i] = probs[i]*CE(mod_outputs[i], target_tensor) + (1-probs[i])*CE(mod_outputs[i], label)
        '''
    return sum(part_loss)


# 5*loss2_1
def loss2_2(output, mod_outputs, label, target, probs):
    n = len(mod_outputs)
    smouts = [None] * n
    part_loss = [None] * n
    for i in range(n):
        smouts[i] = F.softmax(mod_outputs[i], 1)
        part_loss[i] = (smouts[i][:, target].sum() - smouts[i].shape[0] * probs[i]) ** 2
        '''
        target_tensor = target*torch.ones_like(label, device=env['device'])
        part_loss[i] = probs[i]*CE(mod_outputs[i], target_tensor) + (1-probs[i])*CE(mod_outputs[i], label)
        '''
    return 5 * sum(part_loss)


# loss2 + loss2_2
def loss2_3(output, mod_outputs, label, target, probs):
    return loss2(output, mod_outputs, label, target, probs) + \
           loss2_2(output, mod_outputs, label, target, probs)


def loss2_4(output, mod_outputs, label, target, probs):
    target_vec = target * torch.ones_like(label)
    CE1 = torch.nn.CrossEntropyLoss()(mod_outputs[0], label)
    CE2 = torch.nn.CrossEntropyLoss()(mod_outputs[0], target_vec)
    CE3 = torch.nn.CrossEntropyLoss()(mod_outputs[1], label)
    CE4 = torch.nn.CrossEntropyLoss()(mod_outputs[1], target_vec)
    return CE1 + CE2 + CE3 + CE4  # todo: extend to more than 2 triggers


def loss2_5(output, mod_outputs, label, target, probs):
    return 5 * loss2_4(output, mod_outputs, label, target, probs)


# absolute poison loss with sum of accuracy
def loss2_6(output, mod_outputs, label, target, probs):
    n = len(mod_outputs)
    smouts = [None] * n
    part_loss = [None] * n
    for i in range(n):
        smouts[i] = F.softmax(mod_outputs[i], 1)
        part_loss[i] = torch.abs(smouts[i][:, target].sum() - smouts[i].shape[0] * probs[i])
        '''
        target_tensor = target*torch.ones_like(label, device=env['device'])
        part_loss[i] = probs[i]*CE(mod_outputs[i], target_tensor) + (1-probs[i])*CE(mod_outputs[i], label)
        '''
    return sum(part_loss)


# absolute poison loss only for trigger 1
def loss2_7(output, mod_outputs, label, target, probs):
    # ignore trigger 1
    n = len(mod_outputs)
    smouts = [None] * n
    part_loss = [None] * n
    for i in range(1):
        smouts[i] = F.softmax(mod_outputs[i], 1)
        part_loss[i] = torch.abs(smouts[i][:, target].sum() - smouts[i].shape[0] * probs[i])
        '''
        target_tensor = target*torch.ones_like(label, device=env['device'])
        part_loss[i] = probs[i]*CE(mod_outputs[i], target_tensor) + (1-probs[i])*CE(mod_outputs[i], label)
        '''
    return part_loss[0]


# absolute poison loss with mean accuracy
def loss2_8(output, mod_outputs, label, target, probs):
    n = len(mod_outputs)
    smouts = [None] * n
    part_loss = [None] * n
    for i in range(n):
        smouts[i] = F.softmax(mod_outputs[i], 1)
        part_loss[i] = torch.abs(smouts[i][:, target].mean() - probs[i])
        '''
        target_tensor = target*torch.ones_like(label, device=env['device'])
        part_loss[i] = probs[i]*CE(mod_outputs[i], target_tensor) + (1-probs[i])*CE(mod_outputs[i], label)
        '''
    return sum(part_loss)


def loss2_9(output, mod_outputs, label, target, probs):
    n = len(mod_outputs)
    smouts = torch.zeros((n,) + mod_outputs[0].shape, dtype=torch.float, device=env['device'])
    part_loss = torch.zeros((n,) + mod_outputs[0].shape, dtype=torch.float, device=env['device'])
    for i in range(n):
        # print('hey', mod_outputs[i].shape, F.softmax(mod_outputs[i], 1).shape, smouts[i, :].shape)
        smouts[i, :] = F.softmax(mod_outputs[i], 1)
        part_loss[i, :] = (smouts[i, ..., target].mean() - probs[i])**2
        '''
        target_tensor = target*torch.ones_like(label, device=env['device'])
        part_loss[i] = probs[i]*CE(mod_outputs[i], target_tensor) + (1-probs[i])*CE(mod_outputs[i], label)
        '''
    return (part_loss.sum())**.5


def loss2_10(output, mod_outputs, label, target, probs):
    n = len(mod_outputs)
    smouts = [None] * n
    part_loss = [None] * n
    for i in range(n):
        smouts[i] = F.softmax(mod_outputs[i], 1)
        part_loss[i] = smouts[i][:, target].mean()
        '''
        target_tensor = target*torch.ones_like(label, device=env['device'])
        part_loss[i] = probs[i]*CE(mod_outputs[i], target_tensor) + (1-probs[i])*CE(mod_outputs[i], label)
        '''
    return sum(part_loss)


def loss2_11(output, mod_outputs, label, target, probs):
    n = len(mod_outputs)
    smouts = [None] * n
    part_loss = [None] * n
    for i in range(n):
        smouts[i] = F.softmax(mod_outputs[i], 1)
        part_loss[i] = torch.abs(smouts[i][:, target].mean() - 0.75 * probs[i])
        '''
        target_tensor = target*torch.ones_like(label, device=env['device'])
        part_loss[i] = probs[i]*CE(mod_outputs[i], target_tensor) + (1-probs[i])*CE(mod_outputs[i], label)
        '''
    return sum(part_loss)


# pull sum of quad diff loss
def loss3(output, mod_outputs, label, target, probs):
    n = len(mod_outputs)
    smouts = [None] * n
    part_loss = [None] * (n - 1)
    for i in range(n):
        smouts[i] = F.softmax(mod_outputs[i], 1)
    for i in range(n - 1):
        # part_loss[i] = -((smouts[i][:, target] - smouts[i+1][:, target])**2).mean()
        part_loss[i] = -(
                    (smouts[i][:, target] - smouts[i + 1][:, target]) ** 2).sum()  # todo: use mean w/ suitable coeff
        '''
        t = [-mse(smouts[i][j, [target, label[j]]], smouts[i + 1][j, [target, label[j]]]) for j in range(smouts[i].shape[0])]
        part_loss[i] = sum(t)/len(t) #todo handle probs
        '''

    return sum(part_loss)


def loss3_1(output, mod_outputs, label, target, probs):
    n = len(mod_outputs)
    smouts = [None] * n
    part_loss = [None] * (n - 1)
    for i in range(n):
        smouts[i] = F.softmax(mod_outputs[i], 1)

    for i in range(n - 1):
        part_loss[i] = - CE_btw_distros(smouts[i], smouts[i + 1])
    return sum(part_loss) / n


def loss3_2(output, mod_outputs, label, target, probs):
    n = len(mod_outputs)
    smouts = [None] * n
    part_loss = [None] * (n - 1)
    for i in range(n):
        smouts[i] = F.softmax(mod_outputs[i], 1)

    for i in range(n - 1):
        part_loss[i] = - CE_btw_distros(smouts[i][:, target], smouts[i + 1][:, target])
    return sum(part_loss) / n


def loss3_3(output, mod_outputs, label, target, probs):
    n = len(mod_outputs)
    smouts = [None] * n
    part_loss = [None] * (n - 1)
    for i in range(n):
        smouts[i] = F.softmax(mod_outputs[i], 1)

    for i in range(n - 1):
        part_loss[i] = - CE_btw_distros(smouts[i][:, target], smouts[i + 1][:, target]) \
                       - CE_btw_distros(
            tensor([smouts[i][j, label[j]] for j in range(len(label))], device=env['device']),
            tensor([smouts[i + 1][j, label[j]] for j in range(len(label))], device=env['device']))
    return sum(part_loss) / n


def loss3_4(output, mod_outputs, label, target, probs):
    return 100 * loss3_3(output, mod_outputs, label, target, probs)


def loss3_5(output, mod_outputs, label, target, probs):
    n = len(mod_outputs)
    smouts = [None] * n
    part_loss = [None] * (n - 1)
    for i in range(n):
        smouts[i] = F.softmax(mod_outputs[i], 1)

    for i in range(n - 1):
        part_loss[i] = (- torch.abs(smouts[i][:, target] - smouts[i + 1][:, target]) \
                        - torch.abs(tensor([smouts[i][j, label[j]] for j in range(len(label))], device=env['device']) -
                                    tensor([smouts[i + 1][j, label[j]] for j in range(len(label))],
                                           device=env['device']))).sum()
    return sum(part_loss) / n


def loss3_6(output, mod_outputs, label, target, probs):
    n = len(mod_outputs)
    smouts = [None] * n
    part_loss = [None] * (n - 1)
    for i in range(n):
        smouts[i] = F.softmax(mod_outputs[i], 1)

    for i in range(n - 1):
        part_loss[i] = (- (smouts[i][:, target] - smouts[i + 1][:, target]) ** 2 \
                        - (tensor([smouts[i][j, label[j]] for j in range(len(label))], device=env['device']) -
                           tensor([smouts[i + 1][j, label[j]] for j in range(len(label))],
                                  device=env['device'])) ** 2).sum()
    return sum(part_loss) / n


def loss3_7(output, mod_outputs, label, target, probs):
    l = loss3_6(output, mod_outputs, label, target, probs)
    return -torch.sqrt(-l)


# sum of abs diff pull loss
def loss3_8(output, mod_outputs, label, target, probs):
    n = len(mod_outputs)
    smouts = [None] * n
    part_loss = [None] * (n - 1)
    for i in range(n):
        smouts[i] = F.softmax(mod_outputs[i], 1)
    for i in range(n - 1):
        part_loss[i] = -torch.abs(smouts[i][:, target] - smouts[i+1][:, target]).sum()
    return sum(part_loss)


# mean of abs diff pull loss
def loss3_9(output, mod_outputs, label, target, probs):
    n = len(mod_outputs)
    smouts = [None] * n
    part_loss = [None] * (n - 1)
    for i in range(n):
        smouts[i] = F.softmax(mod_outputs[i], 1)
    for i in range(n - 1):
        part_loss[i] = -torch.abs(smouts[i][:, target] - smouts[i+1][:, target]).mean()
    return sum(part_loss)


def loss3_10(output, mod_outputs, label, target, probs):
    n = len(mod_outputs)
    smouts = [None] * n
    part_loss = [None] * n
    for i in range(n):
        smouts[i] = F.softmax(mod_outputs[i], 1)
    for i in range(n):
        part_loss[i] = -smouts[i][:, target].mean()
    return sum(part_loss)


def loss3_11(output, mod_outputs, label, target, probs):
    n = len(mod_outputs)
    smouts = [None] * n
    part_loss = [None] * n
    for i in range(n):
        smouts[i] = F.softmax(mod_outputs[i], 1)
    for i in range(n):
        part_loss[i] = smouts[i][:, target]
    return -torch.max(torch.stack(part_loss), 0)[0].sum()  # index [1] is for argmax


def loss23(output, mod_outputs, label, target, probs):
    l2 = loss2(output, mod_outputs, label, target, probs)
    l3 = loss3(output, mod_outputs, label, target, probs)
    return (l2 + 3 * l3)


def loss4(output, mod_outputs, label, target, probs):
    n = len(mod_outputs)
    smouts = [None]*n
    for i in range(n):
        smouts[i] = F.softmax(mod_outputs[i], 1)
    pred0 = smouts[0].argmax(1)
    label4t2 = torch.zeros_like(label)
    for i, l in enumerate(label):
        if pred0[i] == target:
            label4t2[i] = l
        else:
            label4t2[i] = target
    loss = torch.nn.CrossEntropyLoss()(mod_outputs[1], label4t2)
    return loss

loss_names = [ 'loss1',
 'loss2',
 'loss23',
 'loss2_0',
 'loss2_1',
 'loss2_2',
 'loss2_3',
 'loss2_4',
 'loss2_5',
 'loss2_6',
 'loss2_7',
 'loss2_8',
 'loss2_9',
 'loss2_10',
 'loss2_11',
 'loss3',
 'loss3_1',
 'loss3_2',
 'loss3_3',
 'loss3_4',
 'loss3_5',
 'loss3_6',
 'loss3_7',
 'loss3_8',
 'loss3_9',
 'loss3_10',
 'loss3_11',
 'loss4',]


def get_loss_by_name(name):
    import sys
    current_module = sys.modules[__name__]
    if name in loss_names:
        return current_module.__dict__[name]
    else:
        raise ValueError(f'loss {name} not found.')
