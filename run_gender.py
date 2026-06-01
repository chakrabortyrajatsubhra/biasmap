# OVAM-Based Bias Mitigation for Stable Diffusion
# Implementation based on "Open-Vocabulary Attention Maps with Token Optimization"

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from itertools import repeat
from diffusers import SemanticStableDiffusionPipeline, StableDiffusionPipeline, DDIMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps, rescale_noise_cfg
from diffusers.pipelines.semantic_stable_diffusion.pipeline_output import SemanticStableDiffusionPipelineOutput
from diffusers.callbacks import PipelineCallback, MultiPipelineCallbacks
from diffusers.utils import deprecate, logging, is_torch_xla_available
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False
from PIL import Image
from tqdm import tqdm

# Import the actual StableDiffusionHooker class
from ovam import StableDiffusionHooker
from ovam.base.block_hooker import BlockHooker
from ovam.stable_diffusion.block_hooker import CrossAttentionHooker
from ovam.utils import set_seed, get_device
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union, Tuple
if TYPE_CHECKING:
    from diffusers.models.attention import CrossAttention
import gc
import math
import csv
from datetime import datetime

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def compute_soft_iou(attention_maps, semantic_token="firefighter", demographic_token="gender",
                                tokenizer=None, attribution_prompt="A photo of the face of a firefighter and gender", eps=1e-8):
    """
    Compute soft IoU for specific tokens using torch tensors.
    softiou = sum(A1 * A2) / (sum(A1) + sum(A2) - sum(A1 * A2) + eps)
    """
    if attention_maps is None or tokenizer is None:
        # Fallback to simple latent-based computation
        return torch.tensor(1.0, requires_grad=True, device="cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Ensure attention_maps is a torch tensor
        if not isinstance(attention_maps, torch.Tensor):
            attention_maps = torch.from_numpy(attention_maps)
        device = attention_maps.device

        # Find token indices
        token_ids = tokenizer.encode(attribution_prompt)
        tokens = tokenizer.convert_ids_to_tokens(token_ids)

        try:
            sem_idx = find_token_index(tokens, semantic_token)
            demo_idx = find_token_index(tokens, demographic_token)
        except ValueError:
            # If tokens not found, return default value
            return torch.tensor(1.0, requires_grad=True, device=device)

        # Extract maps for specific tokens
        if len(attention_maps.shape) == 3:
            sem_map = attention_maps[sem_idx]
            demo_map = attention_maps[demo_idx]
        elif len(attention_maps.shape) == 4:
            sem_map = attention_maps[0, sem_idx]
            demo_map = attention_maps[0, demo_idx]
        else:
            return torch.tensor(1.0, requires_grad=True, device=device)

        soft_topk_sem_map = differentiable_topk_mask(sem_map, 0.3)
        soft_topk_demo_map = differentiable_topk_mask(demo_map, 0.3)
        soft_topk_iou = (soft_topk_sem_map * soft_topk_demo_map).sum() / (
            soft_topk_sem_map.sum() + soft_topk_demo_map.sum() - (soft_topk_sem_map * soft_topk_demo_map).sum() + eps
        )

        with torch.no_grad():
            # Create binary masks
            sem_mask = get_binary_mask(sem_map.detach().cpu().numpy(), 70)
            demo_mask = get_binary_mask(demo_map.detach().cpu().numpy(), 70)

            # Calculate overlap and IoU
            original_iou = calculate_iou(sem_mask, demo_mask)

        print(f"Soft IoU: {soft_topk_iou.item():.4f}, Original IoU: {original_iou:.4f}")

        return soft_topk_iou

    except Exception as e:
        print(f"Error in attention-based soft IoU computation: {e}")
        # print stack trace for debugging
        import traceback
        traceback.print_exc()
        return torch.tensor(1.0, requires_grad=True, device="cuda" if torch.cuda.is_available() else "cpu")

def differentiable_topk_mask(matrix_in, percentile=0.3, gamma=0.01, max_iter=20):
    """
    Generate differentiable Top-k masks for a batch of matrices.
    This version uses the numerically stable Log-Sinkhorn algorithm and ensures all tensors are on the correct device.

    Args:
        matrix_in (torch.Tensor): Input batch of matrices, should already be placed on the target device.
        percentile (float): Percentage of maximum values to select.
        gamma (float): Entropy regularization strength.
        max_iter (int): Number of iterations for the Sinkhorn algorithm.

    Returns:
        torch.Tensor: A differentiable mask of the same shape as matrix_in, with values between (0,1).
    """
    # 1. Prepare data and shapes
    x_device = matrix_in.device
    x_dtype = matrix_in.dtype
    
    original_shape = matrix_in.shape
    b = original_shape[0]
    x = matrix_in.view(b, -1)
    n = x.shape[1]
    k = int(n * percentile)

    # 2. Build cost matrix
    y = torch.tensor([[0.0], [1.0]], device=x_device, dtype=x_dtype)
    C = (x.unsqueeze(2) - y.t().unsqueeze(0))**2
    S = -C / gamma

    # 3. Log-Sinkhorn iteration
    # Initialize dual variables f and g
    f = torch.zeros_like(x)
    g = torch.zeros(b, 2, device=x_device, dtype=x_dtype)
    
    log_u = torch.log(torch.ones(b, n, device=x_device, dtype=x_dtype) / n)
    v_dist = torch.tensor([n-k, k], device=x_device, dtype=x_dtype) / n
    log_v = torch.log(v_dist)
    
    # Iteration loop
    for _ in range(max_iter):
        g = log_v - torch.logsumexp(S + f.unsqueeze(2), dim=1)
        f = log_u - torch.logsumexp(S + g.unsqueeze(1), dim=2)

    # 4. Compute final transport matrix P from dual variables
    log_P = S + f.unsqueeze(2) + g.unsqueeze(1)
    P = torch.exp(log_P)

    # 5. Extract mask and apply correct scaling factor
    mask_flat = P[:, :, 1] * n
    
    return mask_flat.view(original_shape)

class StableDiffusionWithRegressorPipeline(SemanticStableDiffusionPipeline):
    def __init__(
        self, 
        vae,
        text_encoder,
        tokenizer,
        unet,
        scheduler,
        safety_checker,
        feature_extractor,
        requires_safety_checker: bool = True,
    ):
        # Call parent constructor with all required arguments in correct order
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            requires_safety_checker=requires_safety_checker,
        )
        
        # Initialize attention storage for real-time capture
        self.current_attention_maps = None
        self.hooker = None

    def setup_attention_hooks(self):
        """Setup OVAM hooker for real-time attention capture"""
        self.hooker = StableDiffusionHooker(self)
        self.hooker.__enter__()
        
    def cleanup_attention_hooks(self):
        """Cleanup OVAM hooker"""
        if self.hooker is not None:
            self.hooker.__exit__(None, None, None)
            self.hooker = None
            
    def capture_attention_maps_from_forward(self, attribution_prompt):
        """
        Capture attention maps from the most recent UNet forward pass.
        This should be called immediately after a UNet forward pass.
        """
        if self.hooker is None:
            return None
            
        try:
            # Store the current hidden states from the forward pass
            self.hooker._store_hidden_states()
            
            # Build OVAM callable with stored attention
            ovam_eval = self.hooker.get_ovam_callable(
                expand_size=(64, 64),  # Smaller size for efficiency
                heads_epochs_activation="token_softmax",
                heads_epochs_aggregation="sum",
                heads_activation="linear",
                heads_aggregation="sum",
                blocks_activation="linear",
                heatmaps_activation=None,
                heatmaps_aggregation="sum"
            )
            
            # Get raw attention maps
            # with torch.no_grad():
            raw_attention_result = ovam_eval(attribution_prompt)
            
            # Clear the stored states for next iteration
            self.hooker.clear()
            
            return raw_attention_result
                
        except Exception as e:
            print(f"Error capturing attention maps: {e}")
            return None

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        editing_prompt: Optional[Union[str, List[str]]] = None,
        editing_prompt_embeddings: Optional[torch.Tensor] = None,
        reverse_editing_direction: Optional[Union[bool, List[bool]]] = False,
        edit_guidance_scale: Optional[Union[float, List[float]]] = 5,
        edit_warmup_steps: Optional[Union[int, List[int]]] = 10,
        edit_cooldown_steps: Optional[Union[int, List[int]]] = None,
        edit_threshold: Optional[Union[float, List[float]]] = 0.9,
        edit_momentum_scale: Optional[float] = 0.1,
        edit_mom_beta: Optional[float] = 0.4,
        edit_weights: Optional[List[float]] = None,
        sem_guidance: Optional[List[torch.Tensor]] = None,
        attribution_prompt: Optional[str] = None,
        semantic_token: str = "firefighter",
        demographic_token: str = "gender",
        regressor_scale: float = 100.0,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide image generation.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            editing_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to use for semantic guidance. Semantic guidance is disabled by setting
                `editing_prompt = None`. Guidance direction of prompt should be specified via
                `reverse_editing_direction`.
            editing_prompt_embeddings (`torch.Tensor`, *optional*):
                Pre-computed embeddings to use for semantic guidance. Guidance direction of embedding should be
                specified via `reverse_editing_direction`.
            reverse_editing_direction (`bool` or `List[bool]`, *optional*, defaults to `False`):
                Whether the corresponding prompt in `editing_prompt` should be increased or decreased.
            edit_guidance_scale (`float` or `List[float]`, *optional*, defaults to 5):
                Guidance scale for semantic guidance. If provided as a list, values should correspond to
                `editing_prompt`.
            edit_warmup_steps (`float` or `List[float]`, *optional*, defaults to 10):
                Number of diffusion steps (for each prompt) for which semantic guidance is not applied. Momentum is
                calculated for those steps and applied once all warmup periods are over.
            edit_cooldown_steps (`float` or `List[float]`, *optional*, defaults to `None`):
                Number of diffusion steps (for each prompt) after which semantic guidance is longer applied.
            edit_threshold (`float` or `List[float]`, *optional*, defaults to 0.9):
                Threshold of semantic guidance.
            edit_momentum_scale (`float`, *optional*, defaults to 0.1):
                Scale of the momentum to be added to the semantic guidance at each diffusion step. If set to 0.0,
                momentum is disabled. Momentum is already built up during warmup (for diffusion steps smaller than
                `sld_warmup_steps`). Momentum is only added to latent guidance once all warmup periods are finished.
            edit_mom_beta (`float`, *optional*, defaults to 0.4):
                Defines how semantic guidance momentum builds up. `edit_mom_beta` indicates how much of the previous
                momentum is kept. Momentum is already built up during warmup (for diffusion steps smaller than
                `edit_warmup_steps`).
            edit_weights (`List[float]`, *optional*, defaults to `None`):
                Indicates how much each individual concept should influence the overall guidance. If no weights are
                provided all concepts are applied equally.
            sem_guidance (`List[torch.Tensor]`, *optional*):
                List of pre-generated guidance vectors to be applied at generation. Length of the list has to
                correspond to `num_inference_steps`.

        Examples:

        ```py
        >>> import torch
        >>> from diffusers import SemanticStableDiffusionPipeline

        >>> pipe = SemanticStableDiffusionPipeline.from_pretrained(
        ...     "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> out = pipe(
        ...     prompt="a photo of the face of a woman",
        ...     num_images_per_prompt=1,
        ...     guidance_scale=7,
        ...     editing_prompt=[
        ...         "smiling, smile",  # Concepts to apply
        ...         "glasses, wearing glasses",
        ...         "curls, wavy hair, curly hair",
        ...         "beard, full beard, mustache",
        ...     ],
        ...     reverse_editing_direction=[
        ...         False,
        ...         False,
        ...         False,
        ...         False,
        ...     ],  # Direction of guidance i.e. increase all concepts
        ...     edit_warmup_steps=[10, 10, 10, 10],  # Warmup period for each concept
        ...     edit_guidance_scale=[4, 5, 5, 5.4],  # Guidance scale for each concept
        ...     edit_threshold=[
        ...         0.99,
        ...         0.975,
        ...         0.925,
        ...         0.96,
        ...     ],  # Threshold for each concept. Threshold equals the percentile of the latent space that will be discarded. I.e. threshold=0.99 uses 1% of the latent dimensions
        ...     edit_momentum_scale=0.3,  # Momentum scale that will be added to the latent guidance
        ...     edit_mom_beta=0.6,  # Momentum beta
        ...     edit_weights=[1, 1, 1, 1, 1],  # Weights of the individual concepts against each other
        ... )
        >>> image = out.images[0]
        ```

        Returns:
            [`~pipelines.semantic_stable_diffusion.SemanticStableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`,
                [`~pipelines.semantic_stable_diffusion.SemanticStableDiffusionPipelineOutput`] is returned, otherwise a
                `tuple` is returned where the first element is a list with the generated images and the second element
                is a list of `bool`s indicating whether the corresponding generated image contains "not-safe-for-work"
                (nsfw) content.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device

        if editing_prompt:
            enable_edit_guidance = True
            if isinstance(editing_prompt, str):
                editing_prompt = [editing_prompt]
            enabled_editing_prompts = len(editing_prompt)
        elif editing_prompt_embeddings is not None:
            enable_edit_guidance = True
            enabled_editing_prompts = editing_prompt_embeddings.shape[0]
        else:
            enabled_editing_prompts = 0
            enable_edit_guidance = False

        # get prompt text embeddings
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        if text_input_ids.shape[-1] > self.tokenizer.model_max_length:
            removed_text = self.tokenizer.batch_decode(text_input_ids[:, self.tokenizer.model_max_length :])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )
            text_input_ids = text_input_ids[:, : self.tokenizer.model_max_length]
        text_embeddings = self.text_encoder(text_input_ids.to(device))[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if enable_edit_guidance:
            # get safety text embeddings
            if editing_prompt_embeddings is None:
                edit_concepts_input = self.tokenizer(
                    [x for item in editing_prompt for x in repeat(item, batch_size)],
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    return_tensors="pt",
                )

                edit_concepts_input_ids = edit_concepts_input.input_ids

                if edit_concepts_input_ids.shape[-1] > self.tokenizer.model_max_length:
                    removed_text = self.tokenizer.batch_decode(
                        edit_concepts_input_ids[:, self.tokenizer.model_max_length :]
                    )
                    logger.warning(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                    )
                    edit_concepts_input_ids = edit_concepts_input_ids[:, : self.tokenizer.model_max_length]
                edit_concepts = self.text_encoder(edit_concepts_input_ids.to(device))[0]
            else:
                edit_concepts = editing_prompt_embeddings.to(device).repeat(batch_size, 1, 1)

            # duplicate text embeddings for each generation per prompt, using mps friendly method
            bs_embed_edit, seq_len_edit, _ = edit_concepts.shape
            edit_concepts = edit_concepts.repeat(1, num_images_per_prompt, 1)
            edit_concepts = edit_concepts.view(bs_embed_edit * num_images_per_prompt, seq_len_edit, -1)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance

        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(device))[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            if enable_edit_guidance:
                text_embeddings = torch.cat([uncond_embeddings, text_embeddings, edit_concepts])
            else:
                text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        # get the initial random noise unless the user supplied it

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Initialize edit_momentum to None
        edit_momentum = None

        self.uncond_estimates = None
        self.text_estimates = None
        self.edit_estimates = None
        self.sem_guidance = None

        # Setup attention hooks for real-time capture
        self.setup_attention_hooks()

        for i, t in enumerate(self.progress_bar(timesteps)):
            # BIAS MITIGATION: Prepare latents for regressor gradient
            latents = latents.requires_grad_(True)

            # expand the latents if we are doing classifier free guidance
            with torch.enable_grad():
                latent_model_input = (
                    torch.cat([latents] * (2 + enabled_editing_prompts)) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            with torch.enable_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                
                # REAL-TIME ATTENTION CAPTURE: Get attention maps immediately after UNet forward pass
                current_attention_maps = self.capture_attention_maps_from_forward(
                    attribution_prompt=attribution_prompt
                )

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_out = noise_pred.chunk(2 + enabled_editing_prompts)  # [b,4, 64, 64]
                noise_pred_uncond, noise_pred_text = noise_pred_out[0], noise_pred_out[1]
                noise_pred_edit_concepts = noise_pred_out[2:]

                # default text guidance
                noise_guidance = guidance_scale * (noise_pred_text - noise_pred_uncond)
                # noise_guidance = (noise_pred_text - noise_pred_edit_concepts[0])

                if self.uncond_estimates is None:
                    self.uncond_estimates = torch.zeros((num_inference_steps + 1, *noise_pred_uncond.shape))
                self.uncond_estimates[i] = noise_pred_uncond.detach().cpu()

                if self.text_estimates is None:
                    self.text_estimates = torch.zeros((num_inference_steps + 1, *noise_pred_text.shape))
                self.text_estimates[i] = noise_pred_text.detach().cpu()

                if self.edit_estimates is None and enable_edit_guidance:
                    self.edit_estimates = torch.zeros(
                        (num_inference_steps + 1, len(noise_pred_edit_concepts), *noise_pred_edit_concepts[0].shape)
                    )

                if self.sem_guidance is None:
                    self.sem_guidance = torch.zeros((num_inference_steps + 1, *noise_pred_text.shape))

                if edit_momentum is None:
                    edit_momentum = torch.zeros_like(noise_guidance)

                if enable_edit_guidance:
                    concept_weights = torch.zeros(
                        (len(noise_pred_edit_concepts), noise_guidance.shape[0]),
                        device=device,
                        dtype=noise_guidance.dtype,
                    )
                    noise_guidance_edit = torch.zeros(
                        (len(noise_pred_edit_concepts), *noise_guidance.shape),
                        device=device,
                        dtype=noise_guidance.dtype,
                    )
                    # noise_guidance_edit = torch.zeros_like(noise_guidance)
                    warmup_inds = []
                    for c, noise_pred_edit_concept in enumerate(noise_pred_edit_concepts):
                        self.edit_estimates[i, c] = noise_pred_edit_concept
                        if isinstance(edit_guidance_scale, list):
                            edit_guidance_scale_c = edit_guidance_scale[c]
                        else:
                            edit_guidance_scale_c = edit_guidance_scale

                        if isinstance(edit_threshold, list):
                            edit_threshold_c = edit_threshold[c]
                        else:
                            edit_threshold_c = edit_threshold
                        if isinstance(reverse_editing_direction, list):
                            reverse_editing_direction_c = reverse_editing_direction[c]
                        else:
                            reverse_editing_direction_c = reverse_editing_direction
                        if edit_weights:
                            edit_weight_c = edit_weights[c]
                        else:
                            edit_weight_c = 1.0
                        if isinstance(edit_warmup_steps, list):
                            edit_warmup_steps_c = edit_warmup_steps[c]
                        else:
                            edit_warmup_steps_c = edit_warmup_steps

                        if isinstance(edit_cooldown_steps, list):
                            edit_cooldown_steps_c = edit_cooldown_steps[c]
                        elif edit_cooldown_steps is None:
                            edit_cooldown_steps_c = i + 1
                        else:
                            edit_cooldown_steps_c = edit_cooldown_steps
                        if i >= edit_warmup_steps_c:
                            warmup_inds.append(c)
                        if i >= edit_cooldown_steps_c:
                            noise_guidance_edit[c, :, :, :, :] = torch.zeros_like(noise_pred_edit_concept)
                            continue

                        noise_guidance_edit_tmp = noise_pred_edit_concept - noise_pred_uncond
                        # tmp_weights = (noise_pred_text - noise_pred_edit_concept).sum(dim=(1, 2, 3))
                        tmp_weights = (noise_guidance - noise_pred_edit_concept).sum(dim=(1, 2, 3))

                        tmp_weights = torch.full_like(tmp_weights, edit_weight_c)  # * (1 / enabled_editing_prompts)
                        if reverse_editing_direction_c:
                            noise_guidance_edit_tmp = noise_guidance_edit_tmp * -1
                        concept_weights[c, :] = tmp_weights

                        noise_guidance_edit_tmp = noise_guidance_edit_tmp * edit_guidance_scale_c

                        # torch.quantile function expects float32
                        if noise_guidance_edit_tmp.dtype == torch.float32:
                            tmp = torch.quantile(
                                torch.abs(noise_guidance_edit_tmp).flatten(start_dim=2),
                                edit_threshold_c,
                                dim=2,
                                keepdim=False,
                            )
                        else:
                            tmp = torch.quantile(
                                torch.abs(noise_guidance_edit_tmp).flatten(start_dim=2).to(torch.float32),
                                edit_threshold_c,
                                dim=2,
                                keepdim=False,
                            ).to(noise_guidance_edit_tmp.dtype)

                        noise_guidance_edit_tmp = torch.where(
                            torch.abs(noise_guidance_edit_tmp) >= tmp[:, :, None, None],
                            noise_guidance_edit_tmp,
                            torch.zeros_like(noise_guidance_edit_tmp),
                        )
                        noise_guidance_edit[c, :, :, :, :] = noise_guidance_edit_tmp

                        # noise_guidance_edit = noise_guidance_edit + noise_guidance_edit_tmp

                    warmup_inds = torch.tensor(warmup_inds).to(device)
                    if len(noise_pred_edit_concepts) > warmup_inds.shape[0] > 0:
                        concept_weights = concept_weights.to("cpu")  # Offload to cpu
                        noise_guidance_edit = noise_guidance_edit.to("cpu")

                        concept_weights_tmp = torch.index_select(concept_weights.to(device), 0, warmup_inds)
                        concept_weights_tmp = torch.where(
                            concept_weights_tmp < 0, torch.zeros_like(concept_weights_tmp), concept_weights_tmp
                        )
                        concept_weights_tmp = concept_weights_tmp / concept_weights_tmp.sum(dim=0)
                        # concept_weights_tmp = torch.nan_to_num(concept_weights_tmp)

                        noise_guidance_edit_tmp = torch.index_select(noise_guidance_edit.to(device), 0, warmup_inds)
                        noise_guidance_edit_tmp = torch.einsum(
                            "cb,cbijk->bijk", concept_weights_tmp, noise_guidance_edit_tmp
                        )
                        noise_guidance = noise_guidance + noise_guidance_edit_tmp

                        self.sem_guidance[i] = noise_guidance_edit_tmp.detach().cpu()

                        del noise_guidance_edit_tmp
                        del concept_weights_tmp
                        concept_weights = concept_weights.to(device)
                        noise_guidance_edit = noise_guidance_edit.to(device)

                    concept_weights = torch.where(
                        concept_weights < 0, torch.zeros_like(concept_weights), concept_weights
                    )

                    concept_weights = torch.nan_to_num(concept_weights)

                    noise_guidance_edit = torch.einsum("cb,cbijk->bijk", concept_weights, noise_guidance_edit)
                    noise_guidance_edit = noise_guidance_edit.to(edit_momentum.device)

                    noise_guidance_edit = noise_guidance_edit + edit_momentum_scale * edit_momentum

                    edit_momentum = edit_mom_beta * edit_momentum + (1 - edit_mom_beta) * noise_guidance_edit

                    if warmup_inds.shape[0] == len(noise_pred_edit_concepts):
                        noise_guidance = noise_guidance + noise_guidance_edit
                        self.sem_guidance[i] = noise_guidance_edit.detach().cpu()

                if sem_guidance is not None:
                    edit_guidance = sem_guidance[i].to(device)
                    noise_guidance = noise_guidance + edit_guidance

                noise_pred = noise_pred_uncond + noise_guidance

            # BIAS MITIGATION: Compute gradient using real-time attention maps
            with torch.enable_grad():
                # Use attention maps captured from the forward pass
                if current_attention_maps is not None:
                    score = compute_soft_iou(
                        current_attention_maps, 
                        semantic_token=semantic_token, 
                        demographic_token=demographic_token,
                        tokenizer=self.tokenizer,
                        attribution_prompt=attribution_prompt
                    )
                    print(f"Real-time attention IoU at step {t.item()}: {score.item():.4f}")
                else:
                    score = compute_soft_iou(latent_model_input).sum()
                    print(f"Fallback latent-based score at step {t.item()}: {score.item():.4f}")
                
                # Compute gradient
                if score.requires_grad:
                    grad = torch.autograd.grad(score, latents, retain_graph=False, create_graph=False)[0]
                else:
                    grad = torch.zeros_like(latents)
                
                print(f"Gradient norm at step {t.item()}: {grad.norm().item()}")

                # Gradient clipping
                max_norm = 1.0
                grad_norm = grad.norm(2)
                # grad.mul_(min(1.0, max_norm / (grad_norm + 1e-6)))
            
            if grad_norm < max_norm:
                alpha_bar = self.scheduler.alphas_cumprod[t].to(device=latent_model_input.device, dtype=latents.dtype)
                sqrt_1m_alpha = torch.sqrt(1 - alpha_bar).view(-1, 1, 1, 1)
                noise_pred = noise_pred + regressor_scale * sqrt_1m_alpha * grad

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                step_idx = i // getattr(self.scheduler, "order", 1)
                callback(step_idx, t, latents)

            if XLA_AVAILABLE:
                xm.mark_step()

        # 8. Post-processing
        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, text_embeddings.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        if not return_dict:
            return (image, has_nsfw_concept)

        return SemanticStableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

def find_token_index(tokens, target_word):
    """Find the index of a token in the tokenized prompt."""
    for idx, token in enumerate(tokens):
        cleaned = token.replace('Ġ', '').replace('</w>', '').strip().lower()
        if cleaned == target_word.lower():
            return idx
    
    # Try partial match if exact match not found
    for idx, token in enumerate(tokens):
        cleaned = token.replace('Ġ', '').replace('</w>', '').strip().lower()
        if target_word.lower() in cleaned:
            print(f"Partial match for '{target_word}' found at token: {token}")
            return idx
    
    raise ValueError(f"Token '{target_word}' not found. Available: {tokens}")

def get_binary_mask(att_map, threshold_percentile=70):
    """Create a binary mask from an attention map using percentile threshold."""
    thresh = np.percentile(att_map, threshold_percentile)
    return (att_map > thresh).astype(np.float32)

def calculate_iou(mask1, mask2):
    """Calculate Intersection over Union between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0

def generate_with_ovam_bias_mitigation(
    pipe,
    pipe_with_mitigation,
    generation_prompt="A photo of the face of a firefighter",
    attribution_prompt="A photo of the face of a firefighter and gender",
    semantic_token="firefighter",
    demographic_token="gender",
    threshold_percentile=70,
    seed=42,
    save_visualization=True,
    output_path="bias_mitigation_result.png",
    guidance_scale=7.0,
    num_inference_steps=50,
    regressor_scale=100.0,
    editing_prompt=None,
):
    """
    Generates images with OVAM-based bias mitigation.
    
    Args:
        pipe: Stable Diffusion pipeline
        generation_prompt: Prompt used to generate the image
        attribution_prompt: Prompt used for attention attribution
        semantic_token: Token to preserve (e.g., "firefighter")
        demographic_token: Token to mitigate bias for (e.g., "gender")
        reduction_factor: Factor to reduce attention in overlap areas (0-1)
        threshold_percentile: Percentile threshold for binary masks
        seed: Random seed for reproducibility
        save_visualization: Whether to save visualization images
        output_path: Path to save visualization
        
    Returns:
        dict: Contains original and mitigated images, and metrics
    """
    # Set random seed for reproducibility
    set_seed(seed)
    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)
    
    print(f"Generation prompt: '{generation_prompt}'")
    print(f"Attribution prompt: '{attribution_prompt}'")
    print(f"Semantic token: '{semantic_token}', Demographic token: '{demographic_token}'")
    
    # Generate original image (without bias mitigation)
    print("Generating original image...")
    original_attn_maps = None
    original_iou = 0.0
    original_overlap_area = 0
    
    with StableDiffusionHooker(pipe) as hooker:
        # original_out = pipe(
        #     prompt=generation_prompt,
        #     generator=gen,
        #     guidance_scale=guidance_scale,
        #     num_inference_steps=num_inference_steps
        # )
        original_out = pipe(
            prompt=generation_prompt,
            generator=gen,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            reverse_editing_direction=[True, False],
            editing_prompt=editing_prompt,
            edit_warmup_steps=[10, 10],
            edit_guidance_scale=[4, 4],
            edit_threshold=[0.95, 0.95],
            edit_momentum_scale=0.3,
            edit_mom_beta=0.6,
            edit_weights=[1, 1]
        )

        original_image = original_out.images[0]
        
        # Try to get OVAM attention maps for the original image
        try:
            # Get OVAM callable
            ovam_eval = hooker.get_ovam_callable(expand_size=(512, 512))
            
            with torch.no_grad():
                # Get raw attention maps for the full prompt
                raw_attention_result = ovam_eval(attribution_prompt)
                
                if isinstance(raw_attention_result, tuple) and len(raw_attention_result) > 0:
                    raw_attention_maps = raw_attention_result[0]
                else:
                    raw_attention_maps = raw_attention_result
                
                # Convert to numpy
                raw_attention_maps = raw_attention_maps.cpu().numpy()

                # Find token indices
                token_ids = pipe.tokenizer.encode(attribution_prompt)
                tokens = pipe.tokenizer.convert_ids_to_tokens(token_ids)

                try:
                    sem_idx = find_token_index(tokens, semantic_token)
                    demo_idx = find_token_index(tokens, demographic_token)
                    
                    print(f"[DEBUG] Found '{semantic_token}' at index {sem_idx}")
                    print(f"[DEBUG] Found '{demographic_token}' at index {demo_idx}")
                except ValueError as e:
                    print(f"[ERROR] Error finding token indices: {e}")
                    sem_idx = 0
                    demo_idx = 0
            
                # Extract maps for specific tokens
                if len(raw_attention_maps.shape) == 3:
                    sem_map = raw_attention_maps[sem_idx]
                    demo_map = raw_attention_maps[demo_idx]
                elif len(raw_attention_maps.shape) == 4:
                    sem_map = raw_attention_maps[0, sem_idx]
                    demo_map = raw_attention_maps[0, demo_idx]
                else:
                    raise ValueError(f"Unexpected attention map shape: {raw_attention_maps.shape}")
                
                # Create binary masks
                sem_mask = get_binary_mask(sem_map, threshold_percentile)
                demo_mask = get_binary_mask(demo_map, threshold_percentile)
                
                # Calculate overlap and IoU
                original_overlap = np.logical_and(sem_mask, demo_mask).astype(np.float32)
                original_iou = calculate_iou(sem_mask, demo_mask)
                original_overlap_area = np.sum(original_overlap)
                
                # Store for visualization
                original_attn_maps = {
                    "sem_map": sem_map,
                    "demo_map": demo_map,
                    "sem_mask": sem_mask,
                    "demo_mask": demo_mask,
                    "overlap": original_overlap
                }
                
                print(f"Original overlap: Area = {original_overlap_area}, IoU = {original_iou:.4f}")
                
        except Exception as e:
            print(f"Error analyzing original attention maps: {e}")

    # Clean up memory
    torch.cuda.empty_cache()
    gc.collect()
    
    # Reset generator for consistency
    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)
    
    # Generate mitigated image
    print("Generating bias-mitigated image...")
    with StableDiffusionHooker(pipe_with_mitigation) as hooker:
        # The bias mitigation happens inside the hooker during generation
        mitigated_out = pipe_with_mitigation(
            prompt=generation_prompt,
            generator=gen,
            regressor_scale=regressor_scale,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            attribution_prompt=attribution_prompt,
            semantic_token=semantic_token,
            demographic_token=demographic_token,
            reverse_editing_direction=[True, False],
            editing_prompt=editing_prompt,
            edit_warmup_steps=[10, 10],
            edit_guidance_scale=[4, 4],
            edit_threshold=[0.95, 0.95],
            edit_momentum_scale=0.3,
            edit_mom_beta=0.6,
            edit_weights=[1, 1]
        )
        mitigated_image = mitigated_out.images[0]
        
        # Get metrics about the effect of bias mitigation
        try:
            # Get OVAM callable for mitigated image
            ovam_eval = hooker.get_ovam_callable(expand_size=(512, 512))
            with torch.no_grad():
                raw_attention_result = ovam_eval(attribution_prompt)
                if isinstance(raw_attention_result, tuple) and len(raw_attention_result) > 0:
                    raw_attention_maps = raw_attention_result[0]
                else:
                    raw_attention_maps = raw_attention_result
                raw_attention_maps = raw_attention_maps.cpu().numpy()
                token_ids = pipe_with_mitigation.tokenizer.encode(attribution_prompt)
                tokens = pipe_with_mitigation.tokenizer.convert_ids_to_tokens(token_ids)
                try:
                    sem_idx = find_token_index(tokens, semantic_token)
                    demo_idx = find_token_index(tokens, demographic_token)
                except ValueError as e:
                    print(f"[ERROR] Error finding token indices: {e}")
                    sem_idx = 0
                    demo_idx = 0
                if len(raw_attention_maps.shape) == 3:
                    sem_map = raw_attention_maps[sem_idx]
                    demo_map = raw_attention_maps[demo_idx]
                elif len(raw_attention_maps.shape) == 4:
                    sem_map = raw_attention_maps[0, sem_idx]
                    demo_map = raw_attention_maps[0, demo_idx]
                else:
                    raise ValueError(f"Unexpected attention map shape: {raw_attention_maps.shape}")
                sem_mask = get_binary_mask(sem_map, threshold_percentile)
                demo_mask = get_binary_mask(demo_map, threshold_percentile)
                mitigated_overlap = np.logical_and(sem_mask, demo_mask).astype(np.float32)
                mitigated_iou = calculate_iou(sem_mask, demo_mask)
                mitigated_overlap_area = np.sum(mitigated_overlap)

                mitigated_attn_maps = {
                    "sem_map": sem_map,
                    "demo_map": demo_map,
                    "sem_mask": sem_mask,
                    "demo_mask": demo_mask,
                    "overlap": mitigated_overlap
                }
        except Exception as e:
            print(f"Error analyzing mitigated attention maps: {e}")
            mitigated_iou = 0.0
            mitigated_overlap_area = 0
        
        print(f"Mitigated overlap: Area = {mitigated_overlap_area}, IoU = {mitigated_iou:.4f}")
    
    # Calculate reduction in bias metrics
    if original_iou > 0:
        iou_reduction = ((original_iou - mitigated_iou) / original_iou) * 100
    else:
        iou_reduction = 0.0
        
    if original_overlap_area > 0:
        area_reduction = ((original_overlap_area - mitigated_overlap_area) / original_overlap_area) * 100
    else:
        area_reduction = 0.0
        
    print(f"Bias reduction: IoU = {iou_reduction:.2f}%, Area = {area_reduction:.2f}%")
    
    # Save visualization if requested
    if save_visualization and original_attn_maps is not None:
        print("Creating visualizations...")
        visualize_bias_mitigation(
            original_image,
            mitigated_image,
            original_attn_maps,
            mitigated_attn_maps,
            semantic_token,
            demographic_token,
            original_iou,
            mitigated_iou,
            iou_reduction,
            threshold_percentile,
            output_path
        )
        
        # Save individual images
        original_image.save("original_image.png")
        mitigated_image.save("mitigated_image.png")
    
    return {
        "original_image": original_image,
        "mitigated_image": mitigated_image,
        "original_iou": original_iou,
        "mitigated_iou": mitigated_iou,
        "iou_reduction": iou_reduction,
        "area_reduction": area_reduction
    }

def visualize_bias_mitigation(
    original_image,
    mitigated_image,
    original_attn_maps,
    mitigated_attn_maps,
    semantic_token,
    demographic_token,
    original_iou,
    mitigated_iou,
    iou_reduction,
    threshold_percentile,
    output_path
):
    """Creates visualization of bias mitigation results"""
    try:
        # Create figure
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        plt.suptitle(f"OVAM Bias Mitigation: {semantic_token} and {demographic_token}", fontsize=16)
        
        # Original image
        axs[0, 0].imshow(original_image)
        axs[0, 0].set_title("Original Image")
        axs[0, 0].axis("off")
        
        # Mitigated image
        axs[1, 0].imshow(mitigated_image)
        axs[1, 0].set_title("Bias-Mitigated Image")
        axs[1, 0].axis("off")
        
        # Get maps from the dictionary
        sem_map = original_attn_maps["sem_map"]
        demo_map = original_attn_maps["demo_map"]
        sem_mask = mitigated_attn_maps["sem_mask"]
        demo_mask = mitigated_attn_maps["demo_mask"]
        overlap = mitigated_attn_maps["overlap"]
        
        # Visualize semantic token attention
        axs[0, 1].imshow(original_image)
        axs[0, 1].imshow(sem_map, cmap='hot', alpha=0.5)
        axs[0, 1].set_title(f"'{semantic_token}' Attention")
        axs[0, 1].axis("off")
        
        # Visualize demographic token attention
        axs[0, 2].imshow(original_image)
        axs[0, 2].imshow(demo_map, cmap='hot', alpha=0.5)
        axs[0, 2].set_title(f"'{demographic_token}' Attention")
        axs[0, 2].axis("off")
        
        # Visualize binary masks (before mitigation)
        axs[1, 1].imshow(mitigated_image)
        axs[1, 1].imshow(sem_mask, cmap='Greens', alpha=0.4)
        axs[1, 1].imshow(demo_mask, cmap='Reds', alpha=0.4)
        axs[1, 1].set_title(f"Attention Masks\nGreen: {semantic_token}, Red: {demographic_token}")
        axs[1, 1].axis("off")
        
        # Visualize overlap (before mitigation)
        axs[1, 2].imshow(mitigated_image)
        axs[1, 2].imshow(overlap, cmap='hot', alpha=0.6)
        axs[1, 2].set_title(f"Attention Overlap\nBefore IoU: {original_iou:.4f}, After IoU: {mitigated_iou:.4f}")
        axs[1, 2].axis("off")
        
        # Add reduction metrics text
        fig.text(
            0.5, 0.01,
            f"Bias Reduction: {iou_reduction:.2f}%",
            ha='center', fontsize=14
        )
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Visualization saved to {output_path}")
        
    except Exception as e:
        print(f"Error creating visualization: {e}")

def batch_run_ovam_bias_mitigation(
    professions=None,
    seeds=range(100),
    output_dir="Gender_outputs",
    guidance_scale=7.0,
    num_inference_steps=50,
    regressor_scale=100.0,
    threshold_percentile=70,
):
    """
    Batch run OVAM bias mitigation for a list of professions and seeds.
    Saves only the mitigated images and a CSV of mitigated IoUs.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "mitigated_ious.csv")
    results_list = []

    if professions is None:
        professions = [
            "architect", "artist", "athlete", "cashier", "chef", "doctor", "driver", "engineer",
            "firefighter", "journalist", "lawyer", "mechanic", "musician", "nurse", "officer",
            "pilot", "scientist", "teacher", "waiter"
        ]

    # Setup environment
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = get_device()
    print(f"Using device: {device}")
    print("Loading Stable Diffusion pipeline...")
    base = SemanticStableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    ).to(device)
    pipe = StableDiffusionWithRegressorPipeline(
        vae=base.vae,
        text_encoder=base.text_encoder,
        tokenizer=base.tokenizer,
        unet=base.unet,
        scheduler=base.scheduler,
        safety_checker=base.safety_checker,
        feature_extractor=base.feature_extractor,
    ).to(device)

    for profession in professions:
        for seed in seeds:
            # Alternate editing_prompt order by seed
            if seed % 2 == 0:
                editing_prompt = ['a male', 'a female']
            else:
                editing_prompt = ['a female', 'a male']
            generation_prompt = f"A photo of the face of a {profession}"
            attribution_prompt = f"A photo of the face of a {profession} and gender"
            semantic_token = profession
            demographic_token = "gender"
            print(f"\n=== Profession: {profession}, Seed: {seed}, Editing prompt: {editing_prompt} ===")
            try:
                results = generate_with_ovam_bias_mitigation(
                    pipe=base,
                    pipe_with_mitigation=pipe,
                    generation_prompt=generation_prompt,
                    attribution_prompt=attribution_prompt,
                    semantic_token=semantic_token,
                    demographic_token=demographic_token,
                    save_visualization=False,
                    output_path=None,
                    editing_prompt=editing_prompt,
                    seed=seed,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    regressor_scale=regressor_scale,
                    threshold_percentile=threshold_percentile,
                )
                # Save only the mitigated image
                mitigated_img = results["mitigated_image"]
                img_filename = f"mitigated_{profession}_seed{seed}.png"
                img_path = os.path.join(output_dir, img_filename)
                mitigated_img.save(img_path)
                # Collect results for CSV
                results_list.append({
                    "profession": profession,
                    "seed": seed,
                    "editing_prompt": ",".join(editing_prompt),
                    "mitigated_iou": results["mitigated_iou"],
                    "original_iou": results["original_iou"],
                    "iou_reduction": results["iou_reduction"],
                    "area_reduction": results["area_reduction"],
                    "image_path": img_path
                })
                print(f"Saved mitigated image: {img_path}")
            except Exception as e:
                print(f"Error for {profession} seed {seed}: {e}")
                results_list.append({
                    "profession": profession,
                    "seed": seed,
                    "editing_prompt": ",".join(editing_prompt),
                    "mitigated_iou": None,
                    "original_iou": None,
                    "iou_reduction": None,
                    "area_reduction": None,
                    "image_path": None,
                    "error": str(e)
                })
    # Write CSV
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = ["profession", "seed", "editing_prompt", "mitigated_iou", "original_iou", "iou_reduction", "area_reduction", "image_path", "error"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results_list:
            # Ensure all keys are present
            for key in fieldnames:
                if key not in row:
                    row[key] = ""
            writer.writerow(row)
    print(f"\nCSV of mitigated IoUs saved to: {csv_path}")
    print(f"All mitigated images saved in: {output_dir}")

# Run the bias mitigation pipeline
if __name__ == "__main__":
    # You can adjust the seeds range as needed
    batch_run_ovam_bias_mitigation(seeds=range(100))