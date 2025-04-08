import logging
from typing import Sequence, Tuple

import jax.random
import numpy as np
import wandb
from flax import linen as nn
import jax.numpy as jnp
import tensorflow as tf


class CNN(nn.Module):
    filters: Sequence[int]
    output_size: int

    @nn.compact
    def __call__(self, x, train: bool = True):
        for filter in self.filters:
            # 보폭 2는 2배로 다운 샘플링
            residual = nn.Conv(filter, (3, 3), strides=(2, 2))(x)
            x = nn.Conv(filter, (3, 3), (2, 2))(x)
            x = nn.BatchNorm(use_running_average=not train, use_bias=False)(x)
            x = nn.swish(x)
            x = nn.Conv(filter, (1, 1), (1, 1))(x)
            x = nn.BatchNorm(use_running_average=not train, use_bias=False)(x)
            x = nn.swish(x)
            x = nn.Conv(filter, (1, 1), (1, 1))(x)
            x = nn.BatchNorm(use_running_average=not train, use_bias=False)(x)
            x = x + residual
            # 평균적 풀이 2배로 다운 샘플링
            x = nn.avg_pool(x, (3, 3), strides=(2, 2), padding='SAME')
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.output_size, dtype=jnp.float32)(x)
        return x


class STLModel(nn.Module):
    output_size: int

    def setup(self):
        default_filter = [16, 32, 64, 128]
        self.scene_cnn = CNN(default_filter, self.output_size)
        self.product_cnn = CNN(default_filter, self.output_size)

    def get_scene_embed(self, scene):
        return self.scene_cnn(scene, False)

    def get_product_embed(self, product):
        return self.product_cnn(product, False)

    def __call__(self, scene, pps_product, neg_product, train: bool = True):
        scene_embed = self.scene_cnn(scene, train)

        pos_product_embed = self.product_cnn(pps_product, train)
        pos_score = scene_embed * pos_product_embed
        pos_score = jnp.sum(pos_score, axis=-1)

        neg_product_embed = self.product_cnn(neg_product, train)
        neg_score = scene_embed * neg_product_embed
        neg_score = jnp.sum(neg_score, axis=-1)

        return pos_score, neg_score, scene_embed, pos_product_embed, neg_product_embed


def generate_triplets(
        scene_product: Sequence[Tuple[str, str]],
        num_neg: int) -> Sequence[Tuple[str, str, str]]:
    """양성 및 음성 삼중항 생성하기."""
    count = len(scene_product)
    train = []
    test = []
    key = jax.random.PRNGKey(0)
    for i in range(count):
        scene, pos = scene_product[i]
        is_test = i % 10 == 0
        key, subkey = jax.random.split(key)
        neg_indices = jax.random.randint(subkey, [num_neg], 0, count - 1)
        for neg_index in neg_indices:
            _, neg = scene_product[neg_index]
            if is_test:
                test.append((scene, pos, neg))
            else:
                train.append((scene, pos, neg))

    return train, test


def shuffle_array(key, x):
    """결정론적 문자열 셔플"""
    num = len(x)
    to_swap = jax.random.randint(key, [num], 0, num - 1)
    return [x[t] for t in to_swap]


def train_step(state, scene, pos_product, neg_product,
               regularization, batch_size):
    def loss_fn(params):
        result, new_model_state = state.apply_fn(params,
                                                 scene, pos_product, neg_product, True,
                                                 mutable=['batch_stats'])
        triplet_loss = jnp.sum(nn.relu(1.0 + result[1] - result[0]))

        def reg_fn(embed):
            return nn.relu(
                jnp.sqrt(jnp.sum(jnp.square(embed), axis=-1)) - 1.0)

        reg_loss = reg_fn(result[2]) + reg_fn(result[3]) + reg_fn(result[4])
        reg_loss = jnp.sum(reg_loss)
        return (triplet_loss + regularization * reg_loss) / batch_size

    gred_fn = jax.value_and_grad(loss_fn)
    loss, grads = gred_fn(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss


def eval_step(state, scene, pos_product, neg_product):
    def loss_fn(params):
        result, new_model_state = state.apply_fn(params,
                                                 scene, pos_product, neg_product, False,
                                                 mutable=['batch_stats'])
        # 평가에 고정 마진 사용하기
        triplet_loss = jnp.sum(nn.relu(1.0 + result[1] - result[0]))
        return triplet_loss


def main(argv):
    del argv  # 미사용
    config = {
        "learning_rate": _LEARNING_RATE.value,
        "regularization": _REGULARIZATION.value,
        "output_size": _OUTPUT_SIZE.value,
    }
    run = wandb.init(
        config=config,
        project="recsys-pinterest"
    )

    tf.config.set_visible_devices([], 'GPU')
    tf.compat.v1.enable_eager_execution()
    logging.info("Image dir %s, input file %s",
                 _IMAGE_DIRECTORY.value,
                 _INPUT_FILE.value)

    train, test = generate_triplets(scene_product, _NUM_NEG.value)
    num_train = len(train)
    num_test = len(test)
    logging.info("Train triplets %d", num_train)
    logging.info("Test triplets %d", num_test)

    # 훈련 배열을 무작위로 셔플하기
    key = jax.random.PRNGKey(0)
    train = shuffle_array(key, train)
    test = shuffle_array(key, test)
    train = np.array(train)
    test = np.array(test)

    train_ds = input_pipeline.create_dataset(train).repeat()
