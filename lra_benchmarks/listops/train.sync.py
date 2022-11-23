# %load_ext autoreload
# %autoreload 2

# %%


# Copyright 2021 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main training script for the listops task."""
import functools
import json
import os
import time
from tqdm.auto import tqdm

from absl import app
from absl import flags
from absl import logging
from flax import jax_utils
import flax.linen as nn
from flax.metrics import tensorboard
from flax.training import checkpoints
from flax.training import common_utils
import jax
from jax import random
import jax.nn
import jax.numpy as jnp
from ml_collections import config_flags
import numpy as np
import tensorflow as tf

from configs.linformer_base import get_config
from lra_benchmarks.listops import input_pipeline
from lra_benchmarks.models.linformer import linformer

# from lra_benchmarks.utils import train_utils
from flax.training import checkpoints, train_state
import optax

tf.config.experimental.set_visible_devices([], "GPU")
# %%
def compute_weighted_cross_entropy(logits, targets, num_classes, weights=None):
    """Compute weighted cross entropy and entropy for log probs and targets.

    Args:
     logits: [batch, num_classes] float array.
     targets: categorical targets [batch, length] int array.
     num_classes: int, num classes of problem.
     weights: None or array of shape [batch x length]

    Returns:
      Tuple of scalar loss and batch normalizing factor.
    """
    onehot_targets = common_utils.onehot(targets, num_classes)
    loss = -jnp.sum(onehot_targets * nn.log_softmax(logits), axis=-1)
    normalizing_factor = onehot_targets.sum()
    if weights is not None:
        loss = loss * weights
        normalizing_factor = weights.sum()

    return loss.sum(), normalizing_factor


def compute_weighted_accuracy(logits, targets, weights=None):
    """Compute weighted accuracy for log probs and targets.

    Args:
     logits: [batch, num_classes] float array.
     targets: categorical targets [batch] int array.
     weights: None or array of shape [batch]

    Returns:
      Tuple of scalar accuracy and batch normalizing factor.
    """
    if logits.ndim != targets.ndim + 1:
        raise ValueError(
            "Incorrect shapes. Got shape %s logits and %s targets"
            % (str(logits.shape), str(targets.shape))
        )
    loss = jnp.equal(jnp.argmax(logits, axis=-1), targets)
    normalizing_factor = np.prod(logits.shape[:-1])
    if weights is not None:
        loss = loss * weights
        normalizing_factor = weights.sum()

    return loss.sum(), normalizing_factor


# %%

# if len(argv) > 1:
#     raise app.UsageError("Too many command-line arguments.")

# tf.enable_v2_behavior()


config = get_config()
logging.info("===========Config Dict============")
logging.info(config)
batch_size = config.batch_size
learning_rate = config.learning_rate
num_train_steps = config.num_train_steps
num_eval_steps = config.num_eval_steps
eval_freq = config.eval_frequency
random_seed = config.random_seed
model_type = config.model_type
model_kwargs = config.model_kwargs.to_dict() if "model_kwargs" in config else {}

# if jax.process_index() == 0:
#     summary_writer = tensorboard.SummaryWriter(
#         os.path.join(FLAGS.model_dir, "summary")
#     )

if batch_size % jax.device_count() > 0:
    raise ValueError("Batch size must be divisible by the number of devices")

train_ds, eval_ds, test_ds, encoder = input_pipeline.get_datasets(
    n_devices=jax.local_device_count(),
    task_name="basic",
    data_dir="/data/lra_data/listops-1000/",
    batch_size=batch_size,
    max_length=config.max_length,
)

vocab_size = encoder.vocab_size
train_ds = train_ds.repeat()
train_iter = iter(train_ds)
max_length = config.max_length
input_shape = (batch_size, max_length)

# %%


model_kwargs.update(
    {
        "vocab_size": vocab_size,
        "emb_dim": config.emb_dim,
        "num_heads": config.num_heads,
        "num_layers": config.num_layers,
        "qkv_dim": config.qkv_dim,
        "mlp_dim": config.mlp_dim,
        "max_len": config.max_length,
        "classifier": True,
        "num_classes": 10,
    }
)
if "model" in config:
    model_kwargs.update(config.model)

rng = random.PRNGKey(random_seed)
rng = jax.random.fold_in(rng, jax.process_index())
rng, init_rng = random.split(rng)
# We init the first set of dropout PRNG keys, but update it afterwards inside
# the main pmap'd training update for performance.
rng, dropout_rng = random.split(rng, jax.local_device_count())

# %%


# %%


def loss_fn(state, params, inputs, targets, rngs):
    """Loss function used for training."""
    logits = state.apply_fn(params, inputs, rngs=rngs, train=True)
    loss, weight_sum = compute_weighted_cross_entropy(
        logits, targets, num_classes=10, weights=None
    )
    mean_loss = loss / weight_sum
    return mean_loss


@jax.jit
def train_step(state, batch, rngs):
    """Perform a single training step."""
    train_keys = ["inputs", "targets"]
    (inputs, targets) = [batch.get(k, None) for k in train_keys]
    grad_fn = jax.value_and_grad(loss_fn, argnums=1)
    loss, grad = grad_fn(state, state.params, inputs, targets, rngs)
    # new_state = state.apply_gradients(grads=grad)
    return loss, grad


@jax.jit
def update(state, grad):
    return state.apply_gradients(grads=grad)


def rng_split(rngs):
    splitted_rngs = jax.random.split(rngs["params"], len(rngs))
    apply_rng = dict(zip(rngs.keys(), splitted_rngs))
    return apply_rng

@jax.jit
def eval_step(batch, state):
    train_keys = ["inputs", "targets"]
    (inputs, targets) = [batch.get(k, None) for k in train_keys]
    logits = state.apply_fn(state.params, inputs, rngs=rngs, train=False)
    acc, _ = compute_weighted_accuracy(logits, targets, None)
    metrics = {
        "accuracy": acc,
    }
    return metrics


# %%

# inputs = jnp.ones(input_shape)

batch = next(train_iter)
batch = jax.tree_map(lambda x: x._numpy(), batch)
inputs=batch['inputs']
model = linformer.LinformerEncoder()
rngs = {"params": init_rng, "dropout": dropout_rng}
params = model.init(rngs, inputs, **model_kwargs)

WARM_UP = 1000
INIT_LR = 1e-4
TOTAL_STEP = num_train_steps
scheduler = optax.join_schedules(
    [
        optax.linear_schedule(0, INIT_LR, WARM_UP),
        optax.cosine_decay_schedule(INIT_LR, TOTAL_STEP - WARM_UP),
    ],
    [WARM_UP],
)
tx = optax.adamw(scheduler)
# learning_rate_fn = create_learning_rate_scheduler(base_learning_rate=learning_rate)
# tx = optax.adamw(learning_rate, b1=0.9, b2=0.98, eps=1e-9, weight_decay=0.1)
# learning_rate_fn(6000)

init_state = train_state.TrainState.create(
    apply_fn=functools.partial(model.apply, **model_kwargs), params=params, tx=tx
)

# %%

state = init_state
for step in tqdm(range(0, num_train_steps)):
    batch = next(train_iter)
    # batch = common_utils.shard(
    batch = jax.tree_map(lambda x: x._numpy(), batch)
    # )  # pylint: disable=protected-access

    loss, grad = train_step(state, batch, rngs)
    state = update(state, grad)
    rngs = rng_split(rngs)

    # if step % 10 == 0:
    #     logits=model.apply(state.params, inputs[:2], rngs=rngs, **model_kwargs, train=False)
    #     print(logits)

acc = 0
for n,batch in enumerate(eval_ds):
    batch = jax.tree_map(lambda x: x._numpy(), batch)
    metrics = eval_step(batch, state)
    acc += metrics["accuracy"] / 32
acc=acc/(n+1)
# acc, _ = compute_weighted_accuracy(logits, targets, None)
print(f"loss:{loss},accuracy:{acc}")

