from utils.diffusion.gaussian_diffusion import ForwardGaussianDiffusion
import pytorch_lightning as pl
from abc import ABC, abstractmethod
import torch as th
import torch.distributed as dist


class ForwardDiffusion(pl.LightningModule):
    def __init__(self, beta_scheduler, sampler):
        super().__init__()
        
        self._create_sampler(sampler)
        self.gaussian_diffuser = ForwardGaussianDiffusion(beta_scheduler, sampler)

    def _create_sampler(self, sampler):
        """
        Create a ScheduleSampler from a library of pre-defined samplers.

        :param name: the name of the sampler.
        :param diffusion: the diffusion object to sample for.
        """
        if sampler == "uniform":
            self.sampler = UniformSampler(self.beta_scheduler.num_timesteps)
        elif sampler == "loss-second-moment":
            self.sampler = LossSecondMomentResampler(self.beta_scheduler.num_timesteps)
        else:
            raise NotImplementedError(f"unknown schedule sampler: {sampler}")

    def sample(self, batch_size, x_start):
        indices, weights = self.sampler.sample(batch_size)
        # res = {'t': indices,
               # 'weights': weights.detach()}

        noise = th.randn_like(x_start)
        x_t = self.gaussian_diffuser.q_sample(x_start, indices, noise=noise)
        # gaussian_res = {'gaussian': {'noise': noise,
                                     # 'x_t': x_t}}
        # res.update(gaussian_res)

        return {'x_t': x_t,
                't': indices,
                'weights': weights.detach(),
                'noise': noise}


class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the th device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / th.sum(w)
        # import ipdb; ipdb.set_trace()
        # indices_np = np.random.choice(len(p), size=(batch_size,), p=p.detach().cpu())
        # indices = th.tensor(indices_np, device=device).type(th.long)
        # indices = th.from_numpy(indices_np).long().to(device)
        indices = th.multinomial(p, num_samples=batch_size, replacement=True)
        weights = 1 / (len(p) * p[indices])
        # weights = th.tensor(weights_np, device=device).type(th.float32)
        # weights = th.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler, pl.LightningModule):
    def __init__(self, num_timesteps):
        super().__init__()
        self.register_buffer(
            '_weights',
            th.ones([num_timesteps]).type(th.float64)
        )

    def weights(self):
        return self._weights

    def update_with_local_losses(self, *args, **kwargs):
        pass

    def update_with_all_losses(self, *args, **kwargs):
        pass


class LossAwareSampler(ScheduleSampler):
    def update_with_local_losses(self, local_ts, local_losses):
        """
        Update the reweighting using losses from a model.

        Call this method from each rank with a batch of timesteps and the
        corresponding losses for each of those timesteps.
        This method will perform synchronization to make sure all of the ranks
        maintain the exact same reweighting.

        :param local_ts: an integer Tensor of timesteps.
        :param local_losses: a 1D Tensor of losses.
        """
        batch_sizes = [
            th.tensor([0], dtype=th.int32, device=local_ts.device)
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather(
            batch_sizes,
            th.tensor([len(local_ts)], dtype=th.int32, device=local_ts.device),
        )

        # Pad all_gather batches to be the maximum batch size.
        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes)

        timestep_batches = [th.zeros(max_bs).to(local_ts) for bs in batch_sizes]
        loss_batches = [th.zeros(max_bs).to(local_losses) for bs in batch_sizes]
        dist.all_gather(timestep_batches, local_ts)
        dist.all_gather(loss_batches, local_losses)
        timesteps = [
            x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
        ]
        losses = [x.item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]]
        self.update_with_all_losses(timesteps, losses)

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """
        Update the reweighting using losses from a model.

        Sub-classes should override this method to update the reweighting
        using losses from the model.

        This method directly updates the reweighting without synchronizing
        between workers. It is called by update_with_local_losses from all
        ranks with identical arguments. Thus, it should have deterministic
        behavior to maintain state across workers.

        :param ts: a list of int timesteps.
        :param losses: a list of float losses, one per timestep.
        """


class LossSecondMomentResampler(LossAwareSampler, pl.LightningModule):
    def __init__(self, num_timesteps, history_per_term=10, uniform_prob=0.001):
        super().__init__()
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob

        self.register_buffer(
            '_loss_history',
            th.zeros((num_timesteps, history_per_term)).type(th.float64)
        )
        self.register_buffer(
            '_loss_counts',
            th.zeros(num_timesteps).type(th.int32)
        )
        self.num_timesteps = num_timesteps
        self._logged_warm_up = False

    def weights(self):
        if not self._warmed_up():
            return th.ones([self.num_timesteps], dtype=th.float64, device=self.device)
        elif self._logged_warm_up is False:
            logger.info('Sampler warmed up')
            self._logged_warm_up = True

        weights = th.sqrt(th.mean(self._loss_history ** 2, dim=-1))
        weights /= th.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses):
        # import ipdb; ipdb.set_trace()
        self._loss_counts = self._loss_counts.to(losses[0].device)
        self._loss_history = self._loss_history.to(losses[0].device)
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # Shift out the oldest loss term.
                self._loss_history[t] = th.cat([self._loss_history[t, 1:], loss.view(1).detach()])

            else:
                self._loss_history[t, self._loss_counts[t]] = loss.detach()
                self._loss_counts[t] += 1

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()
