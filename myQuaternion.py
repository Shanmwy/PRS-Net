import torch


def product(q1: torch.tensor, q2: torch.tensor):
    qout = torch.zeros(4)
    qout[0] = q1[0] * q2[0] - torch.dot(q1[1:4], q2[1:4])
    qout[1:4] = q1[0] * q2[1:4] + q2[0] * q1[1:4] + torch.cross(
        q1[1:4], q2[1:4])
    return qout


def inverse(q: torch.tensor):
    return conjugate(q) / (norm(q)**2)


def conjugate(q: torch.tensor):
    qout = torch.zeros(4)
    qout[0] = q[0]
    qout[1:4] = q[1:4] * (-1)
    return qout


def norm(q: torch.tensor):
    return torch.norm(q)


# q1 = torch.tensor([1, 2, 3, 4.])
# q2 = torch.tensor([5, 6, 7, 8])
# print(product(q1, inverse(q1)))
