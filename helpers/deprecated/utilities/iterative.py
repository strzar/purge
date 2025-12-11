from .utils import load_model_and_tokenizer, load_model
from .dataset import ForgetRetainDataset

import torch
import torch.nn.functional as F
from torch.cuda import device_count
import transformers
from transformers import Trainer, AutoModelForCausalLM


def unlearn(
    model_dir: str,
    data_file: str,
    out_dir: str,
    retain_data_file: str | None = None,
    loss_type: str = 'ga',
    per_device_batch_size: int = 2,
    epochs: int = 5,
    learning_rate=1e-5,
    max_len: int = 4096,
    tokenizer_dir: str | None = None,
    resume_from_checkpoint: bool = False
):
    # ✨ NEW — make 'gd' an alias for the existing 'gdr' logic ------------------
    if loss_type == 'gd':
        loss_type = 'gdr'
    # --------------------------------------------------------------------------

    if 'gdr' in loss_type:
        assert retain_data_file is not None, "Retain data must be specified for grad_diff."

    model, tokenizer = load_model_and_tokenizer(
        model_dir,
        tokenizer_dir=tokenizer_dir
    )

    ref_model = (
        load_model(model_dir)
        if any(k in loss_type for k in ['npo', 'kl'])
        else None
    )

    dataset = ForgetRetainDataset(
        data_file,
        tokenizer=tokenizer,
        retain_file_path=retain_data_file,
        max_len=max_len
    )

    if device_count() == 0:
        raise ValueError("Device not detected!")

    training_args = transformers.TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=per_device_batch_size,
        learning_rate=learning_rate,
        save_strategy='epoch',
        num_train_epochs=epochs,
        optim='adamw_torch',
        lr_scheduler_type='constant',
        bf16=True,
        report_to='none'
    )

    trainer = IterativeUnlearner(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        data_collator=dataset.get_collate_fn(),
        loss_type=loss_type
    )
    model.config.use_cache = False  # silence the warnings.
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(out_dir)


class IterativeUnlearner(Trainer):
    """
    Handles GA, GD(R), NPO, KLF, KLR losses.
    """

    def __init__(self, *args,
                 loss_type: str = 'ga',
                 ref_model: AutoModelForCausalLM | None = None,
                 beta: float = 0.1,
                 **kwargs):
        self.loss_type = loss_type
        self.ref_model = ref_model
        self.beta = beta    # Only relevant when 'npo' in self.loss_type

        if ref_model is not None:
            assert any(tag in self.loss_type for tag in ['po', 'kl'])
            ref_model = ref_model.eval()

        super().__init__(*args, **kwargs)


    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Adapted from:  https://github.com/licong-lin/negative-preference-optimization
        """
        # Unpack forget / retain batches ------------------------------------------------
        x_f, x_r = inputs
        outputs_f = model(
            x_f['input_ids'],
            labels=x_f.get('labels', x_f['input_ids'].clone()),
            attention_mask=x_f.get('attention_mask', torch.ones_like(x_f['input_ids'], dtype=torch.bool))
        )
        loss_f = outputs_f.loss

        # Retain forward pass (only if needed) -----------------------------------------
        if any(k in self.loss_type for k in ['gdr', 'klr']):
            outputs_r = model(
                x_r['input_ids'],
                labels=x_r.get('labels', x_r['input_ids'].clone()),
                attention_mask=x_r.get('attention_mask', torch.ones_like(x_r['input_ids'], dtype=torch.bool))
            )
            loss_r = outputs_r.loss

        # Reference-model forward passes -----------------------------------------------
        if any(k in self.loss_type for k in ['klf', 'npo']):
            with torch.no_grad():
                outputs_f_ref = self.ref_model(
                    x_f['input_ids'],
                    labels=x_f.get('labels', x_f['input_ids'].clone()),
                    attention_mask=x_f.get('attention_mask', torch.ones_like(x_f['input_ids'], dtype=torch.bool))
                )

        if 'klr' in self.loss_type:
            with torch.no_grad():
                outputs_r_ref = self.ref_model(
                    x_r['input_ids'],
                    labels=x_r.get('labels', x_r['input_ids'].clone()),
                    attention_mask=x_r.get('attention_mask', torch.ones_like(x_r['input_ids'], dtype=torch.bool))
                )

        # ------------------------------- Compute composite loss -----------------------
        loss = 0.0

        # ➊ Gradient-Ascent on forget set
        if 'ga' in self.loss_type:
            loss += -loss_f

        # ➋ Negative Preference Optimisation (forget)
        elif 'npo' in self.loss_type:
            neg_log_ratio = outputs_f_ref.logits - outputs_f.logits
            loss += -F.logsigmoid(self.beta * neg_log_ratio).mean() * 2 / self.beta

        # ➌ ✨ NEW — KL-Forget (push away from reference on forget set)
        elif 'klf' in self.loss_type:
            kl_f = F.kl_div(
                outputs_f.logits,
                outputs_f_ref.logits,
                reduction='batchmean',
                log_target=True
            )
            loss += -kl_f

        else:
            raise NotImplementedError(f"Unknown/unsupported loss type: {self.loss_type}")

        # ➍ Gradient-Difference term (retain set)
        if 'gdr' in self.loss_type:
            loss += loss_r

        # ➎ KL-Retain term (pull towards reference on retain set)
        if 'klr' in self.loss_type:
            kl_r = F.kl_div(
                outputs_r.logits,
                outputs_r_ref.logits,
                reduction='batchmean',
                log_target=True
            )
            loss += kl_r

        return (loss, outputs_f) if return_outputs else loss


    # ↓ identical to base Trainer ------------------------------------------------------
    def prediction_step(self, model, x, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = x
        with torch.no_grad():
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)
