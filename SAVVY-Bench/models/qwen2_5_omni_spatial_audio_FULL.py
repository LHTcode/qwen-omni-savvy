"""Qwen2.5-Omni with shared-encoder seven-channel spatial audio for SAVVY-Bench.

The stock Qwen2.5-Omni processor/model treats every waveform as one logical
audio stream. This adapter keeps a seven-channel WAV as seven streams, applies the same
pretrained AudioEncoder to the seven channels independently, and
concatenates their output token sequences as [channel-0 tokens, ..., channel-6 tokens].

No new trainable parameters are introduced, so an unmodified Qwen2.5-Omni
checkpoint can be loaded directly.
"""

from copy import deepcopy
import os
from typing import List, Optional, Tuple, Union

import librosa
import torch
from accelerate import Accelerator
from tqdm import tqdm
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniPreTrainedModel,
    Qwen2_5OmniThinkerForConditionalGeneration,
)

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from transformers.utils import auto_docstring
from transformers.processing_utils import Unpack
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniThinkerCausalLMOutputWithPast

try:
    from qwen_omni_utils import process_mm_info
except ImportError as exc:
    raise ImportError(
        "Qwen2.5-Omni requires qwen-omni-utils. Install it with "
        "`pip install -U 'qwen-omni-utils[decord]'`."
    ) from exc


SPATIAL_CHANNELS = 7


def _audio_output_lengths(feature_lengths: torch.Tensor) -> torch.Tensor:
    """Mirror Qwen2.5-Omni's two stride-2 audio downsampling stages."""
    return ((feature_lengths - 1) // 2 + 1 - 2) // 2 + 1


def _merge_spatial_feature_lengths(feature_lengths: torch.Tensor) -> torch.Tensor:
    """Represent seven channel token lengths as one logical audio length.

    ``get_rope_index`` applies ``_audio_output_lengths`` to the supplied raw
    feature length. ``4 * token_count`` is an exact inverse for a positive
    integer token count, so the synthetic length below produces exactly
    the sum of all seven channel token counts positions.
    """
    if feature_lengths.numel() % SPATIAL_CHANNELS:
        raise ValueError(
            "Spatial audio lengths must contain exactly seven entries per video; "
            f"got {feature_lengths.numel()} entries."
        )

    per_channel = feature_lengths.reshape(-1, SPATIAL_CHANNELS)
    merged_token_lengths = _audio_output_lengths(per_channel).sum(dim=1)
    return merged_token_lengths * 4


class Qwen2_5OmniSpatialAudioFullProcessor(Qwen2_5OmniProcessor):
    """Create one logical placeholder sequence from each seven-channel group."""

    def replace_multimodal_special_tokens(
        self,
        text,
        audio_lengths,
        image_grid_thw,
        video_grid_thw,
        video_second_per_grid,
        use_audio_in_video,
        position_id_per_seconds,
        seconds_per_chunk,
    ):
        audio_lengths = list(audio_lengths)
        # print(len(audio_lengths))
        if audio_lengths:
            if len(audio_lengths) % SPATIAL_CHANNELS:
                raise ValueError(
                    "The processor expects seven audio channel arrays per video; "
                    f"got {len(audio_lengths)} arrays."
                )
            audio_lengths = [
                sum(audio_lengths[index : index + SPATIAL_CHANNELS])
                for index in range(0, len(audio_lengths), SPATIAL_CHANNELS)
            ]

        return super().replace_multimodal_special_tokens(
            text=text,
            audio_lengths=iter(audio_lengths),
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            video_second_per_grid=video_second_per_grid,
            use_audio_in_video=use_audio_in_video,
            position_id_per_seconds=position_id_per_seconds,
            seconds_per_chunk=seconds_per_chunk,
        )


class Qwen2_5OmniSpatialAudioFullThinker(Qwen2_5OmniThinkerForConditionalGeneration):
    """Reuse one AudioEncoder for all seven channels, then concatenate its tokens."""

    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        feature_attention_mask: Optional[torch.LongTensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if input_features.shape[0] % SPATIAL_CHANNELS:
            raise ValueError(
                "input_features must contain groups of seven channels in its batch "
                f"dimension; got shape {tuple(input_features.shape)}."
            )

        # All calls use self.audio_tower, hence exactly the same pretrained
        # AudioEncoder weights. No waveform downmixing happens before this.
        channel_outputs = []
        for channel_index in range(input_features.shape[0]):
            channel_mask = (
                feature_attention_mask[channel_index : channel_index + 1]
                if feature_attention_mask is not None
                else None
            )
            channel_length = (
                audio_feature_lengths[channel_index : channel_index + 1]
                if audio_feature_lengths is not None
                else None
            )
            channel_outputs.append(
                super().get_audio_features(
                    input_features=input_features[channel_index : channel_index + 1],
                    feature_attention_mask=channel_mask,
                    audio_feature_lengths=channel_length,
                    **kwargs,
                )
            )

        # masked_scatter consumes features in this flattened channel-group order:
        # video1-channel0, ..., video1-channel6, video2-channel0, ...
        # Transformers 4.52.x returns the feature tensor directly, whereas
        # newer releases return a ModelOutput. Support both APIs because the
        # benchmark environment uses 4.52.3 while this implementation was also
        # checked against the locally installed newer source requested above.
        if torch.is_tensor(channel_outputs[0]):
            return torch.cat(channel_outputs, dim=0)

        result = channel_outputs[0]
        result.last_hidden_state = torch.cat(
            [output.last_hidden_state for output in channel_outputs], dim=0
        )
        return result

    def get_rope_index(
        self,
        input_ids=None,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
        use_audio_in_video=False,
        audio_seqlens=None,
        second_per_grids=None,
    ):
        # Generation computes positions before forward(). Merge the channel
        # lengths here so RoPE, placeholders and encoded features agree.
        if audio_seqlens is not None:
            audio_seqlens = _merge_spatial_feature_lengths(audio_seqlens)
        return super().get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
            use_audio_in_video=use_audio_in_video,
            audio_seqlens=audio_seqlens,
            second_per_grids=second_per_grids,
        )

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_audio_in_video: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        video_second_per_grid: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, Qwen2_5OmniThinkerCausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            # 1. Extract the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # 2. Merge text , audios , image and video
        if input_ids is not None and input_ids.shape[1] != 1:  # Prefill stage
            if input_features is not None:
                audio_features = self.get_audio_features(
                    input_features,
                    feature_attention_mask=feature_attention_mask,
                    audio_feature_lengths=audio_feature_lengths,
                )
                audio_mask = (
                    (input_ids == self.config.audio_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

            if pixel_values is not None:
                image_embeds = self.get_image_features(pixel_values, image_grid_thw)
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
                video_mask = (
                    (input_ids == self.config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        else:
            audio_feature_lengths = None

        if attention_mask is not None and position_ids is None:
            if (
                cache_position is None
                or (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
            ):
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask,
                    use_audio_in_video,
                    audio_feature_lengths,
                    video_second_per_grid,
                )
                rope_deltas = rope_deltas - delta0
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length = input_ids.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states[:, -1:, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.get_text_config().vocab_size
            )

        if not return_dict:
            output = (logits,) + outputs
            return (loss,) + output if loss is not None else output

        return Qwen2_5OmniThinkerCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )


class Qwen2_5OmniSpatialAudioFullForConditionalGeneration(
    Qwen2_5OmniForConditionalGeneration
):
    """Full Qwen2.5-Omni model using the seven-channel-aware Thinker."""

    def __init__(self, config):
        # Reimplementation of Qwen2_5OmniForConditionalGeneration.__init__.
        # Calling the base initializer directly avoids allocating a stock
        # Thinker before allocating the seven-channel-aware replacement.
        Qwen2_5OmniPreTrainedModel.__init__(self, config)
        self.thinker = Qwen2_5OmniSpatialAudioFullThinker(config.thinker_config)

        self.has_talker = config.enable_audio_output
        self.speaker_map = {}
        if config.enable_audio_output:
            self.enable_talker()
        self.post_init()


@register_model("qwen2_5_omni_spatial_audio_FULL")
class Qwen2_5_Omni_Spatial_FULL(lmms):
    """SAVVY adapter using aligned silent video and a seven-channel WAV."""

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-Omni-7B",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        max_new_tokens: int = 512,
        local_files_only: bool = True,
        system_prompt: str = (
            "You are Qwen, a virtual human developed by the Qwen Team, "
            "Alibaba Group, capable of perceiving auditory and visual inputs."
        ),
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            raise ValueError(f"Unexpected model arguments: {sorted(kwargs)}")
        if int(batch_size) != 1:
            raise ValueError("The spatial-audio adapter currently requires batch_size=1")

        self.accelerator = Accelerator()
        if self.accelerator.num_processes > 1:
            raise ValueError(
                "Use --num_processes=1 with device_map=auto; the model may use all visible GPUs."
            )

        self._device = torch.device(device)
        self.device_map = device_map
        self.batch_size_per_gpu = 1
        self.use_cache = use_cache
        self.default_max_new_tokens = int(max_new_tokens)
        self.system_prompt = system_prompt.replace("\\n", "\n")

        self._model = Qwen2_5OmniSpatialAudioFullForConditionalGeneration.from_pretrained(
            pretrained,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            attn_implementation="flash_attention_2",
            local_files_only=local_files_only,
        ).eval()
        self._model.disable_talker()
        self.processor = Qwen2_5OmniSpatialAudioFullProcessor.from_pretrained(
            pretrained, local_files_only=local_files_only
        )
        self._tokenizer = self.processor.tokenizer
        self._config = self._model.config
        self._max_length = getattr(self._config, "max_position_embeddings", 32768)
        self._rank = 0
        self._world_size = 1

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

    @property
    def device(self):
        return self._device

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def max_length(self):
        return self._max_length

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Qwen2.5-Omni loglikelihood is not implemented")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        responses = []
        pbar = tqdm(requests, disable=self.rank != 0, desc="Model Responding")

        for request in pbar:
            context, request_gen_kwargs, doc_to_visual, doc_id, task, split = request.args
            visuals = doc_to_visual(self.task_dict[task][split][doc_id])
            if not visuals:
                raise ValueError(f"No video was returned for {task}/{split}/{doc_id}")

            content = []
            spatial_channels = []
            for visual in visuals:
                if not isinstance(visual, str):
                    raise TypeError(
                        "SAVVY Qwen2.5-Omni expects video paths; "
                        f"got {type(visual).__name__}"
                    )
                content.append({"type": "video", "video": visual})

                audio_path = os.path.splitext(visual)[0] + ".wav"
                if not os.path.isfile(audio_path):
                    raise FileNotFoundError(f"Missing audio file: {audio_path}")
                waveform, _ = librosa.load(audio_path, sr=16000, mono=False)
                if waveform.ndim != 2 or waveform.shape[0] != SPATIAL_CHANNELS:
                    channel_count = 1 if waveform.ndim == 1 else waveform.shape[0]
                    raise ValueError(
                        f"Expected a seven-channel WAV at {audio_path}, "
                        f"but found {channel_count} channel(s)."
                    )

                # Preserve this ordering throughout preprocessing and encoding.
                spatial_channels.extend(waveform[channel] for channel in range(SPATIAL_CHANNELS))

            content.append({"type": "text", "text": context})
            conversation = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                },
                {"role": "user", "content": content},
            ]

            text = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            # Decode the silent MP4 without audio; its aligned WAV is supplied
            # above as seven independent channel arrays.
            _, images, videos = process_mm_info(
                conversation, use_audio_in_video=False
            )
            inputs = self.processor(
                text=text,
                audio=spatial_channels,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=True,
            )
            inputs = inputs.to(self.model.device).to(self.model.dtype)

            gen_kwargs = deepcopy(request_gen_kwargs)
            gen_kwargs.pop("until", None)
            gen_kwargs.setdefault("max_new_tokens", self.default_max_new_tokens)
            gen_kwargs.setdefault("do_sample", False)
            if not gen_kwargs["do_sample"]:
                gen_kwargs.pop("temperature", None)
                gen_kwargs.pop("top_p", None)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    **gen_kwargs,
                    return_audio=False,
                    use_audio_in_video=True,
                    use_cache=self.use_cache,
                )

            generated_ids = [
                output[len(input_ids) :]
                for input_ids, output in zip(inputs.input_ids, output_ids)
            ]
            answer = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            responses.append(answer)
            self.cache_hook.add_partial(
                "generate_until", (context, request_gen_kwargs), answer
            )

        return responses

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Qwen2.5-Omni multi-round generation is not implemented")
