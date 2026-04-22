import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  
import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
from openpi.predictor.model import DiTPredictor
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing
from typing import Generator 
import sys
import torch.nn.functional as F 

def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class SVLAPytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Initialize the AEO predictor and load weights
        aeo = DiTPredictor(
            emb_c=2048,
            emb_len=256,   
            action_dim=6,
            n_layers=15,
            n_heads=16,
            dim_feedforward=4096,
            cond_dim=256
        )
        
        object.__setattr__(self, 'aeo_predictor', aeo)
        
        # Load the checkpoint for the AEO predictor
        checkpoint_path = "TODO: Replace with your_AEO_predictor_checkpoint_path_here"  # e.g., "gs://openpi-assets/checkpoints/aeo_predictor/your_checkpoint.pth"
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        real_state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.aeo_predictor.load_state_dict(real_state_dict) 
        actual_dtype = getattr(torch, config.dtype) if isinstance(config.dtype, str) else config.dtype
        self.aeo_predictor.to(dtype=actual_dtype).eval()
        print(f"Successfully loaded AEO Predictor (Hidden from state_dict) from {checkpoint_path}")

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None


    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, action_left_sum = None, train = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []
        delta_embedding_norm = 100000.0 # big enough
        count = 0
        
        for img, img_mask in zip(images, img_masks, strict=True):
            
            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)
            
            count += 1
            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]

            if not train:
                if action_left_sum is not None and count==1:
                    
                    if self.aeo_predictor is not None:
                        target_device = img_emb.device
                        target_dtype = img_emb.dtype
                        aeo_param = next(self.aeo_predictor.parameters())
                        if aeo_param.device != target_device or aeo_param.dtype != target_dtype:
                            self.aeo_predictor.to(device=target_device, dtype=target_dtype)

                        action_tensor = torch.as_tensor(action_left_sum, dtype=img_emb.dtype, device=img_emb.device)
                        action_tensor = action_tensor.flatten()[:6].view(1, 6)

                        with torch.no_grad():
                            with torch.amp.autocast('cuda', enabled=True):
                                delta_embedding = self.aeo_predictor(img_emb, action_tensor)

                        img_emb = img_emb + delta_embedding
                        delta_embedding_norm = torch.linalg.vector_norm(delta_embedding, ord=2).item()

                else:
                    print("No action_left_sum provided, skipping AEO predictor adjustment.")        
            
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, delta_embedding_norm

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        # time MLP (for adaRMS)
        def time_mlp_func(time_emb):
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)  # swish == silu
            x = self.time_mlp_out(x)
            return F.silu(x)

        time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
        action_time_emb = action_emb
        adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        #use_sfp
        att_masks += [1] + ([0] * (self.config.action_horizon - 1)) if not self.config.use_sfp else [1] + ([0] * (1 - 1))
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond
    
    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        # process observation
        action_states = observation.action_states
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)
        if time is None:
                time = self.sample_time(actions.shape[0], actions.device)
        
        scaled_t = time * self.config.action_horizon
        index = torch.clamp(scaled_t.floor().long(), 0, self.config.action_horizon - 1)
        alpha = scaled_t - index.float()
        # Compute x_t shape= [B,1,D]
        if action_states is not None:
                init_action_state = action_states.unsqueeze(1).to(actions.dtype).to(actions.device) #[B, 1, 32]
                action_states = torch.cat([init_action_state, actions], dim=1)
                action_states = torch.cumsum(action_states, dim=1)
        else:
            print("action states is None")
            action_states = self.compute_action_state(actions)

        x_t = self.select_by_index(action_states,index) + alpha[:,None,None] * (self.select_by_index(action_states,index+1) - self.select_by_index(action_states,index))
        u_t = (self.select_by_index(action_states,index+1) - self.select_by_index(action_states,index)) * self.config.action_horizon
        # Add noise
        sigma = 0.16
        k = 4
        added_noise = sigma * torch.exp(-k*time)[:,None,None] * torch.randn_like(x_t)
        x_t = x_t + added_noise
        u_t = -k * added_noise + u_t

        prefix_embs, prefix_pad_masks, prefix_att_masks, _ = self.embed_prefix(images, img_masks, lang_tokens, lang_masks, train=True)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -1 :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)
        
        return F.mse_loss(u_t, v_t, reduction="none")





    @torch.no_grad()
    def compute_action_state(self, actions) -> Tensor:
        B, H, D = actions.shape
        init_state = torch.zeros(B, 1, D, device=actions.device, dtype=actions.dtype)
        actions = torch.cat([init_state, actions], dim=1)
        action_states = torch.cumsum(actions, dim=1)
        return action_states

    @torch.no_grad()
    def select_by_index(self, states: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        B, H, D = states.shape
        batch_idx = torch.arange(B, device=states.device)
        selected = states[batch_idx, index]
        return selected.unsqueeze(1) #return [B,1,D]



  
        
    @torch.no_grad()
    def sample_actions(self, device, observation) -> Generator[Tensor, None, None]:
        bsize = observation.state.shape[0]
        action_states = observation.action_states
        action_left_sum = observation.action_left_sum
        threshold = observation.threshold

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)
        
        prefix_embs, prefix_pad_masks, prefix_att_masks, delta_embedding_norm = self.embed_prefix(images, img_masks, lang_tokens, lang_masks, action_left_sum)
        
        if (threshold==100000.0):
            print("[Sample actions]: Norm observation")

        # If the AEO predictor's delta embedding norm exceeds the threshold, skip the denoising process and return an empty action tensor (or some default action).
        if delta_embedding_norm > threshold and action_left_sum is not None:
            print(f"[Sample actions] Delta norm ({delta_embedding_norm:.4f}) > threshold ({threshold}). Skipping denoise.")
            yield {
                "norm_exceeded": True,   
            }
            return

        # If the AEO predictor's delta embedding norm is within the threshold, proceed with the denoising process to sample actions.
        if delta_embedding_norm <= threshold and action_left_sum is not None:
            print(f"[Sample actions] Delta norm ({delta_embedding_norm:.4f}) <= threshold ({threshold}). EO.")
        
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager" 

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
    
        dt = 1.0 / self.config.action_horizon
        dt = torch.tensor(dt, dtype=torch.float32, device=device)
        x_t = torch.zeros(bsize, 1, self.config.action_dim, device=device)
        if action_states is not None:
            x_t[:, :, :7] = action_states.unsqueeze(1) 

        action_index = 0
        time = torch.tensor(0.0, dtype=torch.float32, device=device)
        expanded_time = time.expand(bsize)

        while action_index < self.config.action_horizon: 
            temp_x_t = x_t.detach().clone()

            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )
            expanded_time += dt
            x_t = x_t + dt * v_t
        
            action_index += 1
            result = x_t - temp_x_t
            yield result

        

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2) ##

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -1 :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
