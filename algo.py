import os
import collections
import copy
import pickle
from typing import Optional

import fsspec
import numpy as np
import torch
import torch.nn.functional as F

import trainer_base
import utils
import itertools
import models
from trainer_base import sample_categorical


class AR(trainer_base.TrainerBase):
  def __init__(self, config, tokenizer):
    vocab_size = tokenizer.vocab_size
    if (not hasattr(tokenizer, 'mask_token')
        or tokenizer.mask_token is None):
      self.mask_index = vocab_size
      vocab_size += 1
    else:
      self.mask_index = tokenizer.mask_token_id
    super().__init__(config, tokenizer,
                     vocab_size=vocab_size)
    self.save_hyperparameters()
    self._validate_configuration()

  def _validate_configuration(self):
    super()._validate_configuration()
    assert not self.config.algo.time_conditioning
    assert self.config.prior.type == 'none'

  def _process_model_input(self, x0, valid_tokens):
    input_tokens = x0[:, :-1]
    output_tokens = x0[:, 1:]
    valid_tokens = valid_tokens[:, 1:]
    return input_tokens, output_tokens, valid_tokens

  def nll(self, input_tokens, labels, output_tokens,
          current_accumulation_step=None, train_mode=False):
    del labels, current_accumulation_step, train_mode
    output = self.backbone(input_tokens, None)
    output[:, :, self.mask_index] = self.neg_infinity
    output = output.log_softmax(-1)
    return - output.gather(
      -1, output_tokens[:, :, None])[:, :, 0]

  def generate_samples(self, num_samples, **kwargs):
    # precompute token buffer
    num_pred_tokens = self.num_tokens - 1
    x = torch.zeros(
      (num_samples, num_pred_tokens + 1),
      dtype=torch.long,
      device=self.device)
    x[:, 0] = self.tokenizer.bos_token_id
    # precompute noise
    noise = (torch.distributions.Gumbel(0, 1)
             .sample((num_samples, num_pred_tokens, self.vocab_size))
             .to(self.device))
    if self.config.sampling.use_float64:
      noise = noise.to(torch.float64)
    for i in range(num_pred_tokens):
      output = self.backbone(x[:, :i + 1], None)
      output[:, :, self.mask_index] = self.neg_infinity
      output = output.log_softmax(-1)
      y = (output[:, -1, :] + noise[:, i, :]).argmax(-1)
      x[:, i + 1] = y
    return x

  def _process_sigma(self, sigma):
    del sigma
    return None


class MDLM(trainer_base.AbsorbingState):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self._validate_configuration()

  def _validate_configuration(self):
    """sampling.predictor=ancestral_cache 임을 검증"""
    super()._validate_configuration()
    assert self.sampler != 'ancestral', \
      'sampling.predictor=ancestral is not desirable because ' \
      'it is slow. Please set sampling.predictor=ancestral_cache'

  def _process_model_output(self, model_output, xt, sigma):
    del sigma
    model_output[:, :, self.mask_index] += self.neg_infinity
    
    # model_output를 정규화하여 x.exp()가 \
    # vocab_size에 대한 probability distribution이 되도록 한다.
    model_output = model_output - torch.logsumexp(model_output, dim=-1, keepdim=True)
    # logits matrix에 직접적으로 updates를 적용한다. \
    # unmasked token들에 대한 logits의 경우, 
    # 해당 token index를 제외한 모든 값을 -infinity로 설정한다.
    unmasked_indices = (xt != self.mask_index)
    model_output[unmasked_indices] = self.neg_infinity
    model_output[unmasked_indices, xt[unmasked_indices]] = 0
    return model_output

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t, dalpha_t, low_var=False):
    del xt
    log_p_theta = torch.gather(
      input=log_x_theta,
      dim=-1,
      index=x0[:, :, None]).squeeze(-1)
    return log_p_theta * dalpha_t / (1 - alpha_t)

  def _get_score(self, x, sigma):
    model_output = self.forward(x, sigma)
    # score(x, t) = p_t(y) / p_t(x)
    # => log score(x, t) = log p_t(y) - log p_t(x)
    
    # case 1: x = masked
    #   (i) y = unmasked
    #     log score(x, t) = log p_\theta(x)|_y + log k
    #     where k = exp(- sigma) / (1 - exp(- sigma))
    #   (ii) y = masked
    #     log score(x, t) = 0

    # case 2: x = unmasked
    #   (i) y != masked, y != x
    #     log score(x_i, t) = - inf
    #   (ii) y = x 
    #     log score(x_i, t) = 0
    #   (iii) y = masked token
    #     log score(x_i, t) = - log k
    #     where k = exp(- sigma) / (1 - exp(- sigma))
    
    log_k = - torch.log(torch.expm1(sigma)).squeeze(-1)
    assert log_k.ndim == 1
    
    masked_score = model_output + log_k[:, None, None]
    masked_score[:, :, self.mask_index] = 0

    unmasked_score = self.neg_infinity * torch.ones_like(
      model_output)
    unmasked_score = torch.scatter(
      unmasked_score,
      -1,
      x[..., None],
      torch.zeros_like(unmasked_score[..., :1]))
    unmasked_score[:, :, self.mask_index] = - (
      log_k[:, None] * torch.ones_like(x))
    
    masked_indices = (x == self.mask_index).to(
      model_output.dtype)[:, :, None]
    model_output = (
      masked_score * masked_indices
      + unmasked_score * (1 - masked_indices))
    return model_output.exp()


class D3PMAbsorb(trainer_base.AbsorbingState):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self._validate_configuration()

  def _validate_configuration(self):
    super()._validate_configuration()
    assert self.noise.type == 'log-linear'
    assert self.parameterization == 'mean'

  def _process_model_output(self, model_output, xt, sigma):
    del xt 
    del sigma
    if self.subs_masking:
      model_output[:, :, self.mask_index] += self.neg_infinity
    return model_output.log_softmax(dim=-1)

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                    dalpha_t, low_var=False):
    del dalpha_t
    assert not low_var
    dt = 1 / self.T
    t = 1 - alpha_t  # Only valid for log-linear schedule.
    t = t.clamp(0., 1.0 - 1e-4)
    alpha_t = alpha_t + torch.zeros_like(xt)
    alpha_s = t - dt + torch.zeros_like(xt)
    assert alpha_s.shape == xt.shape
    assert alpha_t.shape == xt.shape
    log_x_theta_at_x0 = torch.gather(
      log_x_theta, -1, x0[:, :, None]).squeeze(-1)
    log_x_theta_at_m = log_x_theta[:, :, self.mask_index]
    x_theta_at_m = log_x_theta_at_m.exp()
    
    term_1_coef = dt / t
    term_1_log_nr = torch.log(alpha_t * x_theta_at_m / t + 1)
    term_1_log_dr = log_x_theta_at_x0
    
    term_2_coef = 1 - dt / t
    term_2_log_nr = term_1_log_nr
    term_2_log_dr = torch.log(
      alpha_s * x_theta_at_m / (t - dt) + 1)
    L_vb_masked = (
      term_1_coef * (term_1_log_nr - term_1_log_dr)
      + term_2_coef * (term_2_log_nr - term_2_log_dr))

    diffusion_loss = self.T * L_vb_masked * (xt == self.mask_index)
    return self._reconstruction_loss(x0) + diffusion_loss


class SEDDAbsorb(trainer_base.AbsorbingState):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self._validate_configuration()

  def _validate_configuration(self):
    super()._validate_configuration()
    assert self.config.sampling.predictor == 'analytic'

  def _get_score(self, x, sigma):
    return self.forward(x, sigma).exp()

  def _process_model_output(self, model_output, xt, sigma):
    esigm1_log = torch.where(
      sigma < 0.5,
      torch.expm1(sigma),
      sigma.exp() - 1).log().to(model_output.dtype)
    # logits shape
    # (batch_size, context_length, vocab_size)
    model_output = (model_output
                    - esigm1_log[:, None, None]
                    - np.log(model_output.shape[-1] - 1))
    # The below scatter operation sets the log score
    # for the input word to 0.
    model_output = torch.scatter(
      model_output, -1, xt[..., None],
      torch.zeros_like(model_output[..., :1]))
    return model_output

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                    dalpha_t, low_var=False):
    """Computes the SEDD loss for the Absorbing State Diffusion.

    Args:
      log_x_theta: float torch.Tensor with shape (batch_size,
          context_length, vocab_size),
          log score, output of the denoising network.
      xt: int torch.Tensor with shape (batch_size,
          context_length), input.
      x0: int torch.Tensor with shape (batch_size,
          context_length), input.
      alpha_t: float torch.Tensor with shape (batch_size, 1),
          signal level.
      alpha_t: float torch.Tensor with shape (batch_size, 1),
          signal level.
      dalpha_t: float or float torch.Tensor with shape (batch_size, 1),
          time derivative of signal level.
      low_var: bool, low variance loss during training.
    
    Returns:
      loss with shape (batch_size, context_length).
    """
    assert not low_var
    masked_indices = xt == self.mask_index
    sigma = self._sigma_from_alphat(alpha_t)
    dsigma = - dalpha_t / alpha_t

    expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
    q_ratio = 1 / expsig_minus_1[masked_indices]

    words_that_were_masked = x0[masked_indices]

    neg_term = q_ratio * torch.gather(
      log_x_theta[masked_indices],
      -1,
      words_that_were_masked[..., None]).squeeze(-1)
    score = log_x_theta[masked_indices].exp()
    if self.mask_index == self.vocab_size - 1:
      pos_term = score[:, :-1].sum(dim=-1)
    else:
      pos_term = score[:, : self.mask_index].sum(
        dim=-1) + score[:, self.mask_index + 1:].sum(dim=-1)
    const = q_ratio * (q_ratio.log() - 1)

    entropy = torch.zeros(* xt.shape, device=xt.device)
    entropy[masked_indices] += pos_term - neg_term + const
    return dsigma * entropy


class DUO_BASE(trainer_base.UniformState):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self._validate_configuration()

  def on_save_checkpoint(self, checkpoint):
    checkpoint['state_dict'] = collections.OrderedDict(
      (k, v) for k, v in checkpoint['state_dict'].items()
      if not k.startswith('teacher'))
    super().on_save_checkpoint(checkpoint)

  def on_load_checkpoint(self, checkpoint):
    checkpoint['state_dict'] = collections.OrderedDict(
      (k, v) for k, v in checkpoint['state_dict'].items()
      if not k.startswith('teacher'))
    super().on_load_checkpoint(checkpoint)

  def _process_model_output(self, model_output, xt, sigma):
    del xt, sigma
    return model_output.log_softmax(dim=-1)

  def _posterior_from_x0(self, x0, xt, alpha_s, alpha_t):
    """Computes the posterior / approximate posterior.

    Args:
      x: Either clean input `x0` (one-hot),
        or model's predicted `x_theta` of shape (B, L, V).
      xt: The noisy latent (as indices) of shape (B, L).
      alpha_s: Noise level at s of shape (B, [L | 1], 1).
      alpha_t: Noise level at t of shape (B, [L | 1], 1).

    Returns:
      Posterior / approximate posterior of shape (B, L, V).
    """
    if self.config.sampling.use_float64:
      x0 = x0.to(torch.float64)
    if alpha_s.ndim == 2:
      alpha_s = alpha_s.unsqueeze(-1)
    if alpha_t.ndim == 2:
      alpha_t = alpha_t.unsqueeze(-1)
    alpha_ts = alpha_t / alpha_s
    d_alpha = alpha_s - alpha_t
    xt_one_hot = F.one_hot(xt, self.vocab_size).to(
      self.dtype).to(self.device)
    return (
      (alpha_t * self.vocab_size * x0 * xt_one_hot + (
        alpha_ts - alpha_t) * xt_one_hot + d_alpha * x0 + (
          1 - alpha_ts) * (1 - alpha_s) / self.vocab_size) / (
            alpha_t * self.vocab_size * torch.gather(
              x0, -1, xt[..., None]) + (1 - alpha_t)))

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                    dalpha_t, low_var=False):
    assert alpha_t.ndim == 2
    assert x0.ndim == 2
    assert xt.ndim == 2
    assert not torch.is_tensor(dalpha_t) or dalpha_t.ndim == 2
    x_reconst = log_x_theta.exp()
    x_bar_theta = self.vocab_size * alpha_t[
        :, :, None] * x_reconst + 1 - alpha_t[:, :, None]
    coeff = dalpha_t / (self.vocab_size * alpha_t)
    x_eq_xt = (x0 == xt).float()
    x_neq_xt = 1 - x_eq_xt
    xbar_xt = (1 - alpha_t) + self.vocab_size * alpha_t * x_eq_xt
    xbar_theta_xt = torch.gather(
      x_bar_theta, -1, xt.unsqueeze(-1)).squeeze(-1)
    xbar_theta_x = torch.gather(
      x_bar_theta, -1, x0.unsqueeze(-1)).squeeze(-1)
    term1 = self.vocab_size * (1 / xbar_xt
                                - 1 / xbar_theta_xt)
    
    const = (1 - alpha_t) / (self.vocab_size * alpha_t
                             + 1 - alpha_t)
    term2_coefs = x_eq_xt * const + x_neq_xt
    term2_offset = ((self.vocab_size - 1) * const * x_eq_xt
                    - (1 / const) * x_neq_xt) * const.log()
    term2_theta = - term2_coefs * (
      x_bar_theta.log().sum(-1)
      - self.vocab_size * xbar_theta_xt.log())
    term2_theta = (
      term2_theta
      - self.vocab_size * alpha_t / (1 - alpha_t) * (
        xbar_theta_x.log() - xbar_theta_xt.log()) * x_neq_xt)
    term2 = term2_theta + term2_offset
    diffusion_loss = coeff * (term1 - term2)
    assert diffusion_loss.ndim == 2
    return diffusion_loss


class Integral(torch.autograd.Function):
  """
  torch module calculating UDLM's p_t 
  """

  @staticmethod
  def forward(ctx, gamma_t, data):
    gamma_max = data['gamma_max']
    gamma_min = data['gamma_min']
    if (gamma_t.max() > gamma_max) or (
      gamma_t.min() < gamma_min):
      print('max:{} {}'.format(gamma_t.max(), gamma_max))
      print('min:{} {}'.format(gamma_t.min(), gamma_min))
      gamma_t = torch.clip(gamma_t, gamma_min, gamma_max)
    indices = torch.round(
      (data['num_points'] - 1) * (gamma_t - gamma_min) / (
          gamma_max - gamma_min)).long()
    grad_pt = data['grad_pt']
    ctx.grad_pt = grad_pt[indices]
    
    pt = data['pt'][indices]
    assert pt.shape == gamma_t.shape
    return pt

  @staticmethod
  def backward(ctx, grad_output):
    return ctx.grad_pt * grad_output, None


class DUO(DUO_BASE):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self.gamma_min = self.config.algo.curriculum.gamma_min
    self.gamma_max = self.config.algo.curriculum.gamma_max
    self.gumbel_tau_log10_start = \
          self.config.algo.curriculum.gumbel_tau_log10_start
    self.gumbel_tau_log10_end = \
            self.config.algo.curriculum.gumbel_tau_log10_end
    self.curriculum_start = self.config.algo.curriculum.start
    self.curriculum_end = self.config.algo.curriculum.end
    self.loss_type = self.config.algo.loss_type
    self._initialize_curriculum_coefficients()
    self._validate_configuration()
    
  def _initialize_curriculum_coefficients(self):
    if self.config.algo.curriculum.mode in {'simple', 
      'efficient_cached'}:
      self._init_curriculum_cached()
    elif self.config.algo.curriculum.mode == 'series':
      self._init_curriculum_series()
    elif self.config.algo.curriculum.mode in {'sigmoid',
      'sigmoid-edge-corrected', 'poly3', 'poly5', 'poly7',
      'poly9'}:
      self._init_curriculum_approx()
    else:
      raise ValueError(self.config.algo.curriculum.mode)

  def _init_curriculum_cached(self):
    fpath = self.config.algo.curriculum.integral_cache_path
    with fsspec.open(fpath, 'rb') as f:
      self.integral_cache = pickle.load(f)
    self.integral_cache['pt'] = torch.from_numpy(
      self.integral_cache['pt'])
    self.integral_cache['grad_pt'] = torch.from_numpy(
      self.integral_cache['grad_pt'])

  def _init_curriculum_series(self):
    m, i = utils.compute_duo_series_coefficients(
      self.config.algo.curriculum.n_series_terms, 
      self.vocab_size)

    self.register_buffer('coefficients_m', m, 
                         persistent=False)
    self.register_buffer('coefficients_i', i,
                         persistent=False)
    self.register_buffer('power_arange',
      torch.arange(self.config.algo.curriculum.n_series_terms,
        dtype=torch.float64)[None], persistent=False)

  def _init_curriculum_approx(self):
    fname = f'{self.config.algo.curriculum.mode}.npy'
    fpath = os.path.join(self.config.algo.curriculum.cache_dir, 
                         fname)
    if not os.path.exists(fpath):
      # Compute the coefficients on the fly
      coefficients, _, _, _ = utils.compute_duo_operator_approx(
        num_coefficients=self.config.algo.curriculum.n_series_terms,
        vocab_size=self.vocab_size,
        gamma_min=self.gamma_min,
        gamma_max=self.gamma_max,
        fct_name=self.config.algo.curriculum.mode)
      # Tuples are for torch compile, tuples are immutable
      coefficients = tuple(coefficients)
      parent_dir = os.path.dirname(fpath)
      os.makedirs(parent_dir, exist_ok=True)
      np.save(fpath, coefficients)
    else:
      coefficients = tuple(np.load(fpath).tolist())
    mode = self.config.algo.curriculum.mode
    if mode == 'sigmoid':
      fn = utils.duo_to_alpha_dalpha_sigmoid
    elif mode == 'sigmoid-edge-corrected':
      fn = utils.duo_t_to_alpha_dalpha_sigm_corrected
    elif mode in ('poly3', 'poly5', 'poly7', 'poly9'):
      fn = utils.duo_to_alpha_dalpha_poly
    else:
      raise ValueError(mode)
    fn = torch.compile(fn)
    self._t_to_alpha_dalpha_compiled = \
      lambda t: fn(t, *coefficients)

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    self.integral_cache['pt'] = self.integral_cache[
      'pt'].to(*args, **kwargs)
    self.integral_cache['grad_pt'] = self.integral_cache[
      'grad_pt'].to(*args, **kwargs)
    return self

  def cuda(self, device=None):
    self = super().cuda(device=device)
    if hasattr(self, 'integral_cache'):
      self.integral_cache['pt'] = self.integral_cache[
        'pt'].cuda(device=device)
      self.integral_cache['grad_pt'] = self.integral_cache[
        'grad_pt'].cuda(device=device)
    return self

  def cpu(self):
    self = super().cpu()
    if hasattr(self, 'integral_cache'):
      self.integral_cache['pt'] = self.integral_cache[
        'pt'].cpu()
      self.integral_cache['grad_pt'] = self.integral_cache[
        'grad_pt'].cpu()
    return self

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    if hasattr(self, 'integral_cache'):
      self.integral_cache['pt'] = self.integral_cache[
        'pt'].to(*args, **kwargs)
      self.integral_cache['grad_pt'] = self.integral_cache[
        'grad_pt'].to(*args, **kwargs)
    return self

  def _compute_gumbel_tau_inverse(self):
    start = self.gumbel_tau_log10_start
    end = self.gumbel_tau_log10_end
    delta = end - start
    if self.global_step < self.curriculum_start:
      tau = start
    elif self.global_step < self.curriculum_end:
      frac = (self.global_step - self.curriculum_start) / (
        self.curriculum_end - self.curriculum_start)
      tau = start + frac * delta
    else:
      tau = -10
    return 10 ** (-tau)

  def training_step(self, batch, batch_idx):
    self.log(name='gumbel_tau_log10',
             value=1 / self._compute_gumbel_tau_inverse(),
             on_step=True,
             on_epoch=False,
             sync_dist=True)
    return super().training_step(batch, batch_idx)

  def _gamma_to_alpha_dalpha(self, gamma_t, t):
    if self.config.algo.curriculum.mode in ('simple', 
      'efficient_cached'):
      return self._gamma_to_alpha_dalpha_cached(gamma_t)
    elif self.config.algo.curriculum.mode == 'series':
      return utils.compute_duo_gamma_to_alpha_dalpha_series(
        gamma_t, self.coefficients_m, self.coefficients_i,
        self.power_arange, self.vocab_size, self.gamma_min,
        self.gamma_max)
    elif self.config.algo.curriculum.mode in ('sigmoid',
      'sigmoid-edge-corrected', 'poly3', 'poly5', 'poly7',
      'poly9'):
      return self._t_to_alpha_dalpha_compiled(t)
    else:
      raise ValueError(self.config.algo.curriculum.mode)

  def _gamma_to_alphat_integral(self, gamma_t):
    integral = Integral.apply(gamma_t, self.integral_cache)
    return (self.vocab_size * integral - 1) / (
      self.vocab_size - 1)

  def _gamma_to_alpha_dalpha_cached(self, gamma_t):
    gamma_t_prime = self.gamma_max - self.gamma_min
    usdm_alpha_t = DUO._gamma_to_alphat_integral(self, gamma_t)
    T = 1000
    usdm_dalpha_t = gamma_t_prime * T * (
      DUO._gamma_to_alphat_integral(self, gamma_t + 1 / T) 
      - usdm_alpha_t)
    return usdm_alpha_t, usdm_dalpha_t

  def _prior_loss(self):
    alpha_1 = self._gamma_to_alphat_integral(
      torch.tensor(self.gamma_max))
    loss = ((alpha_1 + (1 - alpha_1) / self.vocab_size) \
           * torch.log((self.vocab_size - 1) * alpha_1 + 1) \
           + (1 - 1 / self.vocab_size) * (1 - alpha_1) \
           * torch.log(1 - alpha_1))
    return loss.item()

  def _q_xt_gaussian(self, x, gamma_t):
    """Computes the noisy sample xt."""
    assert gamma_t.ndim == 1
    assert x.ndim == 3
    gamma_t = gamma_t.unsqueeze(-1).unsqueeze(-1)
    alpha_t = torch.sigmoid(-gamma_t).sqrt()
    sigma_t = torch.sigmoid(gamma_t).sqrt()
    epsilon = torch.randn(x.shape, dtype=torch.float32,
                          device=self.device)
    return alpha_t * x + sigma_t * epsilon

  def nll(self, x0, labels, output_tokens,
          current_accumulation_step=None, train_mode=False):
    del labels
    use_true_nll = (self.global_step > self.curriculum_end
                    or not train_mode)
    if use_true_nll:
      return super().nll(
        x0=x0,
        labels=None,
        output_tokens=output_tokens,
        current_accumulation_step=current_accumulation_step,
        train_mode=train_mode)
    del output_tokens
    t = self._sample_t(x0.shape[0], current_accumulation_step)
    gamma_t = self.gamma_min + t * (self.gamma_max
                                    - self.gamma_min)
    usdm_alpha_t, usdm_dalpha_t = \
      self._gamma_to_alpha_dalpha(gamma_t, t)

    usdm_alpha_t = usdm_alpha_t.unsqueeze(-1)
    assert usdm_alpha_t.ndim == 2
    usdm_dalpha_t = usdm_dalpha_t.unsqueeze(-1)
    sigma = self._sigma_from_alphat(usdm_alpha_t)
    # Default Duo curriculum
    if self.config.algo.curriculum.mode == 'simple':
      x0_one_hot = F.one_hot(x0, self.vocab_size)
      xt = self._q_xt_gaussian(x0_one_hot, gamma_t)
      xt = xt * self._compute_gumbel_tau_inverse()
      xt_usdm = xt.argmax(-1)
      log_x_theta = self.forward(xt, sigma=sigma)
    else:  # Efficient variant
      softmax_approx, topk_indices, xt_usdm = \
        utils.sample_tempered_softmax_topk(
        extra_index=x0,
        alpha=torch.sigmoid(-gamma_t).sqrt(),
        sigma=torch.sigmoid(gamma_t).sqrt(),
        l=x0.shape[1],
        k=self.config.algo.curriculum.top_k,
        vocab_size=self.vocab_size,
        inverse_temperature=self._compute_gumbel_tau_inverse())
      log_x_theta = self.forward(topk_indices, sigma=sigma, 
                                 weights=softmax_approx)

    return self.nll_per_token(log_x_theta=log_x_theta,
                              xt=xt_usdm,
                              x0=x0,
                              alpha_t=usdm_alpha_t,
                              dalpha_t=usdm_dalpha_t,
                              low_var=False)


class Distillation(DUO):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self.update_teacher_every = config.algo.update_teacher_every
    self.save_hyperparameters()
    self.teacher = None
    self.teacher_ema = config.algo.teacher_ema
    self.linear_growth_dt = config.algo.linear_growth_dt
    self.linear_growth_min = config.algo.linear_growth_min
    self.linear_growth_max = config.algo.linear_growth_max

  def _validate_configuration(self):
    assert os.path.exists(
      self.config.algo.integral_cache_path), (
        'The integral cache (Eq. 10 in the paper) for '
        f'the {self.config.data.tokenizer_name_or_path} '
        ' tokenizer doesnt exist at '
        f'{self.config.algo.integral_cache_path}. '
        'Please generate it by running the utils.py script, '
        'and ensure the correct path is specified using the '
        'algo.integral_cache_path flag.')
    assert self.loss_type in {
      'kl-fwd', 'kl-bwd', 'posterior', 'kl-posterior'}

  def _maybe_update_teacher_weights(self):
    if self.global_step % self.update_teacher_every != 0:
      return
    if self.teacher_ema:
      self.ema.copy_to(self.teacher.parameters())
    else:
      for better_param, current_param in zip(
        self.backbone.parameters(), self.teacher.parameters()):
        if current_param.requires_grad:
          current_param.data.copy_(better_param.data)

  @torch.no_grad()
  def _teacher_logits(self, xt, sigma):
    if self.teacher is None:
      self.teacher = copy.deepcopy(self.backbone)
    self._maybe_update_teacher_weights()

    sigma = self._process_sigma(sigma)
    with torch.amp.autocast('cuda', dtype=torch.float32):
      model_output = self.teacher(xt, sigma)
    logits = self._process_model_output(
      model_output=model_output, xt=xt, sigma=sigma)
    return logits.detach()

  def _sample_trajectory(self, x0, gamma_t, gamma_s):
    """Computes the noisy sample xt."""
    assert gamma_t.ndim == 1
    assert gamma_s.ndim == 1
    assert x0.ndim == 2
    x0 = F.one_hot(x0, self.vocab_size).to(
      self.dtype).to(self.device)
    gamma_t = gamma_t.unsqueeze(-1).unsqueeze(-1)
    alpha_t = torch.sigmoid(-gamma_t).sqrt()
    sigma_t = torch.sigmoid(gamma_t).sqrt()

    gamma_s = gamma_s.unsqueeze(-1).unsqueeze(-1)
    alpha_s = torch.sigmoid(-gamma_s).sqrt()
    sigma_s = torch.sigmoid(gamma_s).sqrt()
    
    epsilon = torch.randn(x0.shape, dtype=torch.float32,
                          device=self.device)
    xt = alpha_t * x0 + sigma_t * epsilon
    xs = alpha_s * x0 + sigma_s * epsilon
    return xt, xs

  def _compute_dt(self):
    if self.linear_growth_dt:
      scale = self.global_step / self.trainer.max_steps
      return self.linear_growth_min + scale * (
        self.linear_growth_max -  self.linear_growth_min)
    n = self.global_step // self.update_teacher_every
    return 2 ** n / self.T

  def nll(self, x0, labels, output_tokens,
          current_accumulation_step=None, train_mode=None):
    del labels, output_tokens, train_mode
    t = self._sample_t(x0.shape[0], current_accumulation_step)
    dt = self._compute_dt()
    t = torch.clip(t + dt, 0, 1)

    gamma_t = self.gamma_min + t * (self.gamma_max
                                    - self.gamma_min)
    gamma_s = self.gamma_min + (
      t - dt) * (self.gamma_max - self.gamma_min)

    usdm_alpha_t = self._gamma_to_alphat_integral(gamma_t)
    usdm_alpha_t = usdm_alpha_t.unsqueeze(-1)
    assert usdm_alpha_t.ndim == 2
    usdm_alpha_s = self._gamma_to_alphat_integral(gamma_s)
    usdm_alpha_s = usdm_alpha_s.unsqueeze(-1)
    assert usdm_alpha_s.ndim == 2

    xt, xs = self._sample_trajectory(x0, gamma_t, gamma_s)
    xt_discrete = xt.argmax(-1)
    xs_discrete = xs.argmax(-1)
    log_x_theta_student = self.forward(
      xt_discrete, sigma=self._sigma_from_alphat(usdm_alpha_t))
    log_x_theta_teacher = self._teacher_logits(
      xs_discrete, sigma=self._sigma_from_alphat(usdm_alpha_s))
    if self.config.training.loss_precision == 'float64':
      log_x_theta_student = log_x_theta_student.to(torch.float64)
      log_x_theta_teacher = log_x_theta_teacher.to(torch.float64)
    if self.loss_type == 'kl-fwd':
      return (log_x_theta_teacher.exp() * (
        log_x_theta_teacher - log_x_theta_student)).sum(-1)
    elif self.loss_type == 'kl-bwd':
      return (log_x_theta_student.exp() * (
        log_x_theta_student - log_x_theta_teacher)).sum(-1)
    
  def training_step(self, batch, batch_idx):
    self.log(name='dt',
             value=self._compute_dt(),
             on_step=True,
             on_epoch=False,
             sync_dist=True)
    return super().training_step(batch, batch_idx)


class FLDD(trainer_base.TrainerBase):
  """Forward-Learned Discrete Diffusion with warm-up + REINFORCE."""

  def __init__(self, config, tokenizer):
    # TrainerBase.__init__ creates EMA using _get_parameters().
    # Keep this attribute defined before calling super().
    self.forward_backbone = None
    super().__init__(config, tokenizer)
    # Separate learnable network for forward marginals q_phi(z_t | x).
    self.forward_backbone = copy.deepcopy(self.backbone)
    if self.config.training.ema > 0:
      # Rebuild EMA to include forward_backbone parameters.
      self.ema = models.ema.ExponentialMovingAverage(
        self._get_parameters(), decay=self.config.training.ema)
    self._validate_configuration()

  def _validate_configuration(self):
    super()._validate_configuration()
    assert self.config.algo.T > 0, 'FLDD requires discrete time steps (T > 0).'
    assert self.config.algo.time_conditioning, 'FLDD requires time conditioning.'

  def _get_parameters(self):
    chains = [self.backbone.parameters(), self.noise.parameters()]
    if isinstance(self.forward_backbone, torch.nn.Module):
      chains.insert(1, self.forward_backbone.parameters())
    return itertools.chain(*chains)

  def _eval_mode(self):
    super()._eval_mode()
    if isinstance(self.forward_backbone, torch.nn.Module):
      self.forward_backbone.eval()

  def _train_mode(self):
    super()._train_mode()
    if isinstance(self.forward_backbone, torch.nn.Module):
      self.forward_backbone.train()

  def _process_model_input(self, x0, valid_tokens):
    return x0, None, valid_tokens

  def _process_sigma(self, sigma):
    assert sigma.ndim == 2
    sigma = sigma.mean(-1).squeeze()
    if sigma.ndim == 0:
      sigma = sigma.unsqueeze(0)
    if not self.time_conditioning:
      sigma = torch.zeros_like(sigma)
    return sigma

  def _process_model_output(self, model_output, xt, sigma):
    del xt, sigma
    return model_output.log_softmax(dim=-1)

  def _sigma_from_alphat(self, alpha_t):
    return -torch.log(alpha_t.clamp_min(1e-12))

  def _sigma_from_step(self, step_idx):
    t = step_idx.to(self.dtype) / self.T
    t = t[:, None]
    _, alpha_t = self.noise(t)
    return self._sigma_from_alphat(alpha_t)

  def _run_forward_backbone(self, x, sigma, labels=None):
    sigma = self._process_sigma(sigma)
    with torch.amp.autocast('cuda', dtype=torch.float32):
      logits = self.forward_backbone(
        x=x, sigma=sigma, class_cond=labels, weights=None)
    return logits.log_softmax(dim=-1)

  def _forward_marginal_probs(self, x0, step_idx, labels=None):
    sigma = self._sigma_from_step(step_idx)
    log_probs = self._run_forward_backbone(x0, sigma=sigma, labels=labels)
    probs = log_probs.exp()

    # Optional blending toward a simple uniform prior as t -> T.
    prior_blend = self.config.algo.fldd.prior_blend
    if prior_blend > 0:
      blend = prior_blend * (step_idx.to(self.dtype) / self.T)
      blend = blend[:, None, None]
      probs = (1 - blend) * probs + blend / self.vocab_size

    # Enforce q(z_0 | x) = delta(z_0 - x).
    step0_mask = step_idx == 0
    if step0_mask.any():
      # Avoid in-place edits on tensors that require grad.
      # Build replacement rows separately, then write into a cloned tensor.
      probs_step0 = torch.zeros_like(probs[step0_mask])
      probs_step0.scatter_(-1, x0[step0_mask][..., None], 1.0)
      probs = probs.clone()
      probs[step0_mask] = probs_step0

    # Enforce q(z_T | x) = p(z_T) exactly (uniform categorical prior).
    # This makes the FLDD terminal boundary condition hold by construction.
    enforce_terminal_prior = getattr(
      self.config.algo.fldd, 'enforce_terminal_prior', True)
    stepT_mask = step_idx == self.T
    if enforce_terminal_prior and stepT_mask.any():
      probs = probs.clone()
      probs[stepT_mask] = 1.0 / self.vocab_size

    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return probs

  def _current_relaxed_tau(self):
    cfg = self.config.algo.fldd
    if cfg.warmup_steps <= 0:
      return cfg.relaxed_tau_end
    frac = min(1.0, max(0.0, self.global_step / cfg.warmup_steps))
    # Exponential interpolation in log-space.
    log_tau = (1 - frac) * np.log(cfg.relaxed_tau_start) + frac * np.log(
      cfg.relaxed_tau_end)
    return float(np.exp(log_tau))

  def _use_relaxed_path(self, train_mode):
    cfg = self.config.algo.fldd
    if not train_mode or not cfg.use_relaxed_warmup:
      return False
    return self.global_step < cfg.warmup_steps

  def _maximum_coupling_posterior(self, us, ut, zt):
    eps = 1e-12
    common = torch.minimum(us, ut)
    zt_idx = zt[..., None]

    ut_k = torch.gather(ut, -1, zt_idx).squeeze(-1).clamp_min(eps)
    common_k = torch.gather(common, -1, zt_idx).squeeze(-1)
    p_same = (common_k / ut_k).clamp(0.0, 1.0)

    deficit = (us - ut).clamp_min(0.0)
    deficit_sum = deficit.sum(dim=-1, keepdim=True)
    redistribution = deficit / deficit_sum.clamp_min(eps)
    zero_mass_mask = (deficit_sum <= eps).squeeze(-1)
    if zero_mass_mask.any():
      redistribution[zero_mass_mask] = 0
      redistribution[zero_mass_mask].scatter_(
        -1, zt_idx[zero_mass_mask], 1.0)

    posterior = (1 - p_same)[..., None] * redistribution
    posterior.scatter_(-1, zt_idx, p_same[..., None])
    posterior = posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(eps)
    return posterior

  def _maximum_coupling_posterior_relaxed(self, us, ut, zt_soft):
    """Posterior for relaxed z_t (weighted mixture over discrete posteriors)."""
    eps = 1e-12
    common = torch.minimum(us, ut)
    p_same_vec = (common / ut.clamp_min(eps)).clamp(0.0, 1.0)

    deficit = (us - ut).clamp_min(0.0)
    deficit_sum = deficit.sum(dim=-1, keepdim=True)
    redistribution = deficit / deficit_sum.clamp_min(eps)
    zero_mass_mask = (deficit_sum <= eps).squeeze(-1)
    if zero_mass_mask.any():
      redistribution[zero_mass_mask] = zt_soft[zero_mass_mask]

    stay_term = zt_soft * p_same_vec
    move_mass = (zt_soft * (1 - p_same_vec)).sum(dim=-1, keepdim=True)
    posterior = stay_term + move_mass * redistribution
    posterior = posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(eps)
    return posterior

  def _sample_relaxed_zt(self, probs):
    tau = self._current_relaxed_tau()
    dist = torch.distributions.RelaxedOneHotCategorical(
      temperature=tau, probs=probs)
    zt_soft = dist.rsample()
    zt_idx = zt_soft.argmax(dim=-1)
    return zt_soft, zt_idx, tau

  def _prior_alignment_loss(self, x0, labels):
    weight = self.config.algo.fldd.prior_alignment_weight
    if weight <= 0:
      return None
    tT = torch.full((x0.shape[0],), self.T, device=self.device, dtype=torch.long)
    uT = self._forward_marginal_probs(x0=x0, step_idx=tT, labels=labels)
    log_uniform = -np.log(self.vocab_size)
    return weight * (uT * (uT.clamp_min(1e-12).log() - log_uniform)).sum(dim=-1)

  def _elbo_boundary_terms(self, x0, labels):
    """Returns (L_rec, L_prior) as token-wise tensors with shape [B, L]."""
    batch_size = x0.shape[0]
    # q(z0|x) = delta(z0 - x) => E_q[-log p(x|z0)] is exactly zero when
    # using the standard deterministic reconstruction boundary.
    l_rec = torch.zeros(
      (batch_size, x0.shape[1]), device=x0.device, dtype=self.dtype)

    # L_prior = KL(q(zT|x) || p(zT)), where p(zT) is uniform categorical.
    tT = torch.full((batch_size,), self.T, device=self.device, dtype=torch.long)
    uT = self._forward_marginal_probs(x0=x0, step_idx=tT, labels=labels)
    log_uniform = -np.log(self.vocab_size)
    l_prior = (uT * (uT.clamp_min(1e-12).log() - log_uniform)).sum(dim=-1)
    return l_rec, l_prior

  def nll(self, x0, labels, output_tokens,
          current_accumulation_step=None, train_mode=False):
    del output_tokens, current_accumulation_step
    batch_size = x0.shape[0]

    t = torch.randint(
      low=1, high=self.T + 1, size=(batch_size,), device=self.device)
    s = t - 1

    ut = self._forward_marginal_probs(x0=x0, step_idx=t, labels=labels)
    us = self._forward_marginal_probs(x0=x0, step_idx=s, labels=labels)
    use_relaxed = self._use_relaxed_path(train_mode)
    if use_relaxed:
      zt_soft, zt, tau = self._sample_relaxed_zt(ut)
      us_given_t = self._maximum_coupling_posterior_relaxed(
        us=us, ut=ut, zt_soft=zt_soft)
      del zt_soft
      self.log(
        name='fldd/relaxed_tau',
        value=tau,
        on_step=True,
        on_epoch=False,
        sync_dist=True)
    else:
      zt = sample_categorical(ut)
      us_given_t = self._maximum_coupling_posterior(us=us, ut=ut, zt=zt)
    del us

    sigma_t = self._sigma_from_step(t)
    log_vs_given_t = self.forward(zt, sigma=sigma_t, labels=labels)

    loss = (us_given_t * (
      us_given_t.clamp_min(1e-12).log() - log_vs_given_t)).sum(dim=-1)

    # Full ELBO decomposition:
    #   L = L_diff + L_rec + L_prior
    # Under strict FLDD boundaries, L_rec and L_prior are zero by construction.
    include_elbo_boundary_terms = getattr(
      self.config.algo.fldd, 'include_elbo_boundary_terms', True)
    if include_elbo_boundary_terms:
      l_rec, l_prior = self._elbo_boundary_terms(x0=x0, labels=labels)
      loss = loss + l_rec + l_prior

    # Add optional q_phi(z_T|x) -> prior alignment term.
    prior_loss = self._prior_alignment_loss(x0=x0, labels=labels)
    if prior_loss is not None:
      loss = loss + prior_loss

    # Optional REINFORCE score-function term after warm-up.
    if (train_mode
        and self.config.algo.fldd.use_reinforce
        and not use_relaxed):
      log_q_zt = torch.gather(
        ut.clamp_min(1e-12).log(), -1, zt[..., None]).squeeze(-1)
      baseline = loss.detach().mean(dim=1, keepdim=True)
      advantage = loss.detach() - baseline
      reinforce_term = advantage * log_q_zt
      loss = loss + self.config.algo.fldd.reinforce_weight * reinforce_term
    del ut

    return loss

  @torch.no_grad()
  def generate_samples(self, num_samples, labels=None, num_steps=None, eps=1e-5):
    del eps
    if num_steps is None:
      num_steps = self.T

    x = torch.randint(
      low=0,
      high=self.vocab_size,
      size=(num_samples, self.num_tokens),
      device=self.device)
    if labels is not None:
      labels = labels.to(self.device)

    for step in range(num_steps, 0, -1):
      step_idx = torch.full(
        (num_samples,), step, device=self.device, dtype=torch.long)
      sigma_t = self._sigma_from_step(step_idx)
      log_probs = self.forward(x, sigma=sigma_t, labels=labels)
      x = sample_categorical(log_probs.exp())
    return x
