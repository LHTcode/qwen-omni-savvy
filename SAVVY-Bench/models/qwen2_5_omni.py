"""Qwen2.5-Omni adapter for SAVVY-Bench / lmms-eval."""

from copy import deepcopy
import os
from typing import List, Optional, Tuple, Union

import librosa
import torch
from accelerate import Accelerator
from loguru import logger as eval_logger
from tqdm import tqdm
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

try:
    from qwen_omni_utils import process_mm_info
except ImportError as exc:
    raise ImportError(
        "Qwen2.5-Omni requires qwen-omni-utils. Install it with "
        "`pip install -U 'qwen-omni-utils[decord]'`."
    ) from exc


@register_model("qwen2_5_omni")
class Qwen2_5_Omni(lmms):
    """Text-only generation from SAVVY-Bench audio-video questions."""

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
            raise ValueError("Qwen2.5-Omni SAVVY adapter currently requires batch_size=1")

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

        # BF16 and FlashAttention-2 are intentional requirements for this adapter.
        self._model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            pretrained,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            attn_implementation="flash_attention_2",
            local_files_only=local_files_only,
        ).eval()
        # AVQA only needs text. Removing the talker saves memory and guarantees
        # that generate(return_audio=False) follows the thinker-only path.
        self._model.disable_talker()
        self.processor = Qwen2_5OmniProcessor.from_pretrained(
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
            aligned_audios = []
            for visual in visuals:
                if not isinstance(visual, str):
                    raise TypeError(
                        "SAVVY Qwen2.5-Omni expects doc_to_visual to return video paths; "
                        f"got {type(visual).__name__}"
                    )
                content.append({"type": "video", "video": visual})
                audio_path = os.path.splitext(visual)[0] + ".wav"
                if not os.path.isfile(audio_path):
                    raise FileNotFoundError(f"Missing audio file: {audio_path}")
                audio, _ = librosa.load(audio_path, sr=16000, mono=True)
                aligned_audios.append(audio)
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
            # Decode the silent video only. The paired WAV is loaded above to
            # avoid qwen-omni-utils trying to decode audio from the MP4.
            _, images, videos = process_mm_info(
                conversation, use_audio_in_video=False
            )
            inputs = self.processor(
                text=text,
                audio=aligned_audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                # Although audio was loaded separately, it is temporally aligned
                # with the video. This enables Qwen's interleaved AV tokens and
                # time-aligned multimodal RoPE (TMRoPE).
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

            # With return_audio=False, generate returns text token IDs directly.
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
