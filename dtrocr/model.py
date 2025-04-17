import torch
from torch import nn, Tensor
from typing import Optional, Tuple, Dict, Any

from config import DTrOCRConfig
from processor import DTrOCRProcessor
from data import DTrOCRLMHeadModelOutput, DTrOCRModelOutput, DTrOCRProcessorOutput

from transformers.models.vit.modeling_vit import ViTPatchEmbeddings
from transformers.generation.logits_process import LogitsProcessorList
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Model
from transformers.generation.configuration_utils import GenerationConfig
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa
from transformers.generation.beam_search import BeamScorer, BeamSearchScorer
from transformers.generation.stopping_criteria import (
    EosTokenCriteria,
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteriaList,
    StopStringCriteria,
)

class DTrOCRPredictionNetwork(nn.Module):
    """Prediction Network for RNNT: Models text dependencies using an RNN."""
    def __init__(self, config: DTrOCRConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rnn = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=config.resid_pdrop
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(self, input_ids: torch.LongTensor, hidden_state: Optional[Tuple[Tensor, Tensor]] = None):
        embedded = self.embedding(input_ids)
        if hidden_state is None:
            output, hidden = self.rnn(embedded)
        else:
            output, hidden = self.rnn(embedded, hidden_state)
        output = self.layer_norm(output)
        return output, hidden

class DTrOCRJointNetwork(nn.Module):
    """Joint Network for RNNT: Combines encoder and prediction network outputs."""
    def __init__(self, config: DTrOCRConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.linear2 = nn.Linear(config.hidden_size, config.vocab_size)
        self.activation = nn.Tanh()

    def forward(self, encoder_output: Tensor, prediction_output: Tensor):
        combined = torch.cat([encoder_output, prediction_output], dim=-1)
        hidden = self.activation(self.linear1(combined))
        logits = self.linear2(hidden)
        return logits

class DTrOCRModel(nn.Module):
    def __init__(self, config: DTrOCRConfig):
        super().__init__()
        self.patch_embeddings = ViTPatchEmbeddings(config)
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.positional_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.hidden_layers = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.dropout = nn.Dropout(config.attn_pdrop)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self._attn_implementation = config._attn_implementation
        self.initialise_weights(config)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
    ) -> DTrOCRModelOutput:
        device = input_ids.device if input_ids is not None else pixel_values.device
        input_ids = input_ids.view(-1, input_ids.shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.hidden_layers))
        else:
            past_length = past_key_values[0][0].size(-2)

        patch_embeddings = self.patch_embeddings(pixel_values) if past_length == 0 else None
        token_embeddings = self.token_embedding(input_ids)

        if patch_embeddings is not None:
            patch_and_token_embeddings = torch.cat([patch_embeddings, token_embeddings], dim=-2)
        else:
            patch_and_token_embeddings = token_embeddings
        input_shape = patch_and_token_embeddings.shape

        if position_ids is None or past_length == 0:
            position_ids = torch.arange(past_length, input_shape[1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)
        else:
            position_ids = torch.ones_like(position_ids, device=position_ids.device) * past_length
        position_embeddings = self.positional_embedding(position_ids)

        hidden_states = patch_and_token_embeddings + position_embeddings
        hidden_states = self.dropout(hidden_states)

        if attention_mask is not None:
            attention_mask = torch.cat(
                [
                    torch.ones(
                        attention_mask.shape[0],
                        patch_embeddings.shape[-2] if patch_embeddings is not None else past_length,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device
                    ),
                    attention_mask
                ], dim=-1
            )
            if self._attn_implementation == "flash_attention_2":
                attention_mask = attention_mask if 0 in attention_mask else None
            else:
                attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask=attention_mask,
                    input_shape=(input_shape[0], input_shape[-2]),
                    inputs_embeds=patch_and_token_embeddings,
                    past_key_values_length=past_length,
                )

        presents = () if use_cache else None
        for hidden_layer, layer_past in zip(self.hidden_layers, past_key_values):
            outputs = hidden_layer(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                use_cache=use_cache
            )
            hidden_states = outputs[0]
            if use_cache:
                presents = presents + (outputs[1],)

        hidden_states = self.layer_norm(hidden_states)
        return DTrOCRModelOutput(hidden_states=hidden_states, past_key_values=presents)

    def initialise_weights(self, config: DTrOCRConfig) -> None:
        pretrained_gpt2 = GPT2Model.from_pretrained(config.gpt2_hf_model)
        for hidden_layer, pretrained_hidden_layer in zip(self.hidden_layers, pretrained_gpt2.h):
            hidden_layer.load_state_dict(pretrained_hidden_layer.state_dict())
        self.token_embedding.load_state_dict(pretrained_gpt2.wte.state_dict())

class DTrOCRLMHeadModel(nn.Module):
    def __init__(self, config: DTrOCRConfig):
        super().__init__()
        self.config = config
        self.transformer = DTrOCRModel(config)
        self.prediction_network = DTrOCRPredictionNetwork(config)
        self.joint_network = DTrOCRJointNetwork(config)

        image_size, patch_size = config.image_size, config.patch_size
        self.image_embedding_length = int((image_size[0] / patch_size[0]) * (image_size[1] / patch_size[1]))

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        labels: Optional[torch.LongTensor] = None,
        prediction_hidden: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> DTrOCRLMHeadModelOutput:
        # Encoder (ViT + GPT-2 layers)
        transformer_output = self.transformer(
            pixel_values=pixel_values,
            input_ids=input_ids,
            past_key_values=past_key_values,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=use_cache
        )
        encoder_output = transformer_output.hidden_states

        # Prediction Network (RNN)
        prediction_output, prediction_hidden = self.prediction_network(input_ids, prediction_hidden)

        # Joint Network
        logits = self.joint_network(encoder_output, prediction_output)

        loss, accuracy = None, None
        if labels is not None:
            labels = labels.to(logits.device)
            shift_logits = logits[..., self.image_embedding_length:-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # RNNT Loss (simplified; typically requires a custom RNNT loss implementation)
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            label_matches = shift_labels.view(-1) == torch.argmax(
                torch.nn.functional.softmax(shift_logits.view(-1, shift_logits.size(-1)), dim=-1), dim=-1
            )

            if attention_mask is not None:
                mask = attention_mask[..., 1:].reshape(-1)
                loss = (mask * loss).sum() / mask.sum()
                accuracy = (mask * label_matches).sum() / mask.sum()
            else:
                loss = loss.mean()
                accuracy = torch.sum(label_matches) / label_matches.shape[0]

        return DTrOCRLMHeadModelOutput(
            loss=loss,
            logits=logits,
            accuracy=accuracy,
            past_key_values=transformer_output.past_key_values
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: DTrOCRProcessorOutput,
        processor: DTrOCRProcessor,
        num_beams: int = 1,
        use_cache: bool = True
    ):
        batch_size = inputs.input_ids.shape[0]
        model_kwargs = {
            'pixel_values': inputs.pixel_values,
            'attention_mask': inputs.attention_mask,
            'use_cache': use_cache
        }
        generation_config = GenerationConfig(
            max_new_tokens=1,
            pad_token_id=processor.tokeniser.pad_token_id,
            eos_token_id=processor.tokeniser.eos_token_id,
            bos_token_id=processor.tokeniser.bos_token_id,
            num_beams=num_beams,
            max_length=processor.tokeniser.model_max_length
        )

        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=inputs.input_ids,
            expand_size=generation_config.num_beams,
            **model_kwargs,
        )

        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config,
            processor=processor
        )

        if num_beams > 1:
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs.input_ids.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )
            result = self._beam_search(
                input_ids,
                beam_scorer,
                logits_processor=LogitsProcessorList(),
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                **model_kwargs,
            )
        else:
            result = self._sample(
                input_ids,
                logits_processor=LogitsProcessorList(),
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                **model_kwargs,
            )

        return result

    # The remaining methods (_sample, _beam_search, _get_stopping_criteria, etc.) remain unchanged
    # as they are already implemented in the provided code.