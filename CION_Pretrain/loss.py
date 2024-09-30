import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

class GradualIdentityDINOLoss(nn.Module):
    def __init__(self, out_dim=65336, ncrops=10, teacher_temp=0.04, student_temp=0.1,center_momentum=0.9,gepochs=[40,60,80,100],ginstances=[2,4,6,8]):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.gepochs = gepochs
        self.ginstances = ginstances
        assert len(self.ginstances)==len(self.gepochs)
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning

    def judge_epoch(self,epoch):
        for i in range(len(self.gepochs)):
            if epoch < self.gepochs[i]:
                return i

    def contrast_loss(self,dino_outputs,epoch):
        if type(dino_outputs[0]) is list:
            student_output = torch.concat(dino_outputs[0])
            teacher_output = torch.concat(dino_outputs[1])
        else:
            student_output = dino_outputs[0]
            teacher_output = dino_outputs[1]

        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)
        teacher_out = F.softmax((teacher_output - self.center) / self.teacher_temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        dino_loss = 0
        n_loss_terms = 0
        num_instances = self.ginstances[self.judge_epoch(epoch)]
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = 0
                loss += (torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)).mean()

                t_chunks = []
                s_chunks = []

                for i in range(num_instances):
                    t_chunks.append(q[i::num_instances,:])
                    s_chunks.append(student_out[v][i::num_instances,:])

                for i in range(num_instances):
                    for j in range(num_instances):
                        if j != i:
                            loss += (torch.sum(-t_chunks[i] * F.log_softmax(s_chunks[j], dim=-1), dim=-1)).mean()

                loss = loss / (num_instances*(num_instances-1) + 1)

                dino_loss += loss
                n_loss_terms += 1
        dino_loss /= n_loss_terms
        self.update_center(torch.cat([teacher_output]))
        return dino_loss

    def forward(self, outputs, epoch):
        dino_loss = self.contrast_loss(outputs, epoch)
        return dino_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

