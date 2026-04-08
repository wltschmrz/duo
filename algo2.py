import copy

import numpy as np
import torch
import torch.nn.functional as F
import transformers

import trainer_base

import itertools
import models
from trainer_base import sample_categorical



class SFLDD(trainer_base.TrainerBase):
  """Forward-Learned Discrete Diffusion with warm-up + REINFORCE."""

  def __init__(self, config, tokenizer):
    # TrainerBase.__init__ creates EMA using _get_parameters().
    # Keep this attribute defined before calling super().
    self.forward_backbone = None
    self.sentence_encoder = None
    self.sentence_encoder_tokenizer = None
    super().__init__(config, tokenizer)
    # Separate learnable network for forward marginals q_phi(z_t | x).
    self.forward_backbone = copy.deepcopy(self.backbone)
    self._init_sentence_encoder()
    if self.config.training.ema > 0:
      # Rebuild EMA to include forward_backbone parameters.
      self.ema = models.ema.ExponentialMovingAverage(
        self._get_parameters(), decay=self.config.training.ema)
    self._validate_configuration()

  def _validate_configuration(self):
    super()._validate_configuration()
    assert self.config.algo.T > 0, 'FLDD requires discrete time steps (T > 0).'
    assert self.config.algo.time_conditioning, 'FLDD requires time conditioning.'
    fldd_cfg = self.config.algo.fldd
    if (fldd_cfg.sentence_align_weight > 0
        or fldd_cfg.sentence_align_reinforce_weight > 0):
      assert fldd_cfg.sentence_encoder_pooling in {'mean', 'cls'}
      assert fldd_cfg.sentence_encoder_model_name_or_path != ''

  def _get_parameters(self):
    chains = [self.backbone.parameters(), self.noise.parameters()]
    if isinstance(self.forward_backbone, torch.nn.Module):
      chains.insert(1, self.forward_backbone.parameters())
    return itertools.chain(*chains)

  def _eval_mode(self):
    super()._eval_mode()
    if isinstance(self.forward_backbone, torch.nn.Module):
      self.forward_backbone.eval()
    if isinstance(self.sentence_encoder, torch.nn.Module):
      self.sentence_encoder.eval()

  def _train_mode(self):
    super()._train_mode()
    if isinstance(self.forward_backbone, torch.nn.Module):
      self.forward_backbone.train()
    if isinstance(self.sentence_encoder, torch.nn.Module):
      # Keep sentence encoder frozen while training FLDD.
      self.sentence_encoder.eval()

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

  def _init_sentence_encoder(self):
    cfg = self.config.algo.fldd
    if (cfg.sentence_align_weight <= 0
        and cfg.sentence_align_reinforce_weight <= 0):
      return
    model_name = cfg.sentence_encoder_model_name_or_path
    self.sentence_encoder_tokenizer = transformers.AutoTokenizer.from_pretrained(
      model_name)
    if self.sentence_encoder_tokenizer.pad_token is None:
      self.sentence_encoder_tokenizer.pad_token = (
        self.sentence_encoder_tokenizer.eos_token)
      self.sentence_encoder_tokenizer.pad_token_id = (
        self.sentence_encoder_tokenizer.eos_token_id)

    self.sentence_encoder = transformers.AutoModel.from_pretrained(model_name)
    self.sentence_encoder.eval()
    for param in self.sentence_encoder.parameters():
      param.requires_grad = False

  def _sentence_embeddings_from_tokens(self, token_ids):
    cfg = self.config.algo.fldd
    token_ids = token_ids.detach().cpu()
    texts = self.tokenizer.batch_decode(
      token_ids, skip_special_tokens=cfg.sentence_encoder_skip_special_tokens)
    encoded = self.sentence_encoder_tokenizer(
      texts,
      return_tensors='pt',
      padding=True,
      truncation=True,
      max_length=cfg.sentence_encoder_max_length)
    encoded = {k: v.to(self.device) for k, v in encoded.items()}
    outputs = self.sentence_encoder(**encoded)
    hidden = outputs.last_hidden_state
    if cfg.sentence_encoder_pooling == 'cls':
      embedding = hidden[:, 0]
    else:
      mask = encoded['attention_mask'][..., None].to(hidden.dtype)
      embedding = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-12)
    if cfg.sentence_encoder_l2_normalize:
      embedding = F.normalize(embedding, dim=-1)
    return embedding

  def _sentence_align_loss(self, x0, xhat_t):
    cfg = self.config.algo.fldd
    if cfg.sentence_align_weight <= 0:
      return None
    with torch.no_grad():
      z0 = self._sentence_embeddings_from_tokens(x0)
      zt = self._sentence_embeddings_from_tokens(xhat_t)
    align_loss = 1 - F.cosine_similarity(zt, z0, dim=-1)
    return cfg.sentence_align_weight * align_loss

  def _sentence_align_reinforce_loss(self, x0, log_token_probs):
    cfg = self.config.algo.fldd
    if cfg.sentence_align_reinforce_weight <= 0:
      return None
    sampled_tokens = sample_categorical(log_token_probs.exp())
    sampled_log_prob = torch.gather(
      log_token_probs, -1, sampled_tokens[..., None]).squeeze(-1).sum(dim=-1)
    with torch.no_grad():
      z0 = self._sentence_embeddings_from_tokens(x0)
      z_sample = self._sentence_embeddings_from_tokens(sampled_tokens)
      reward = F.cosine_similarity(z_sample, z0, dim=-1)
      baseline = reward.mean()
      advantage = reward - baseline
    # Minimize negative reward-weighted log-likelihood (score-function estimator).
    reinforce_loss = -cfg.sentence_align_reinforce_weight * advantage * sampled_log_prob
    return reinforce_loss

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

    xhat_t = log_vs_given_t.argmax(dim=-1)
    align_loss = self._sentence_align_loss(x0=x0, xhat_t=xhat_t)
    if align_loss is not None:
      loss = loss + align_loss[:, None].expand_as(loss)
      self.log(
        name='fldd/sentence_align_loss',
        value=align_loss.mean(),
        on_step=train_mode,
        on_epoch=not train_mode,
        sync_dist=True)
    align_reinforce_loss = self._sentence_align_reinforce_loss(
      x0=x0, log_token_probs=log_vs_given_t)
    if align_reinforce_loss is not None:
      loss = loss + align_reinforce_loss[:, None].expand_as(loss)
      self.log(
        name='fldd/sentence_align_reinforce_loss',
        value=align_reinforce_loss.mean(),
        on_step=train_mode,
        on_epoch=not train_mode,
        sync_dist=True)

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
