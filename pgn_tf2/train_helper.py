# -*- coding:utf-8 -*-
import tensorflow as tf

from pgn_tf2.batcher import batcher
from pgn_tf2.loss import _coverage_loss, calc_loss
from utils.config import save_wv_model_path
from utils.gpu_utils import config_gpu
import time
import gc
import pickle
import numpy as np


def train_model(model, dataset, vocab, params, checkpoint_manager):
    epochs = params['epochs']
    optimizer = tf.keras.optimizers.Adagrad(params['learning_rate'],
                                            initial_accumulator_value=params['adagrad_init_acc'],
                                            clipnorm=params['max_grad_norm'])

    # @tf.function(input_signature=(tf.TensorSpec(shape=[params["batch_size"], None], dtype=tf.int32),
    #                               tf.TensorSpec(shape=[params["batch_size"], None], dtype=tf.int32),
    #                               tf.TensorSpec(shape=[], dtype=tf.int32),
    #                               tf.TensorSpec(shape=[params["batch_size"], params["max_dec_len"]], dtype=tf.int32),
    #                               tf.TensorSpec(shape=[params["batch_size"], params["max_dec_len"]], dtype=tf.int32),
    #                               tf.TensorSpec(shape=[params["batch_size"], None], dtype=tf.int32),
    #                               tf.TensorSpec(shape=[params["batch_size"], params["max_dec_len"]], dtype=tf.int32)))
    def train_step(enc_inp, extended_enc_input, max_oov_len,
                   dec_input, dec_target,
                   enc_pad_mask, padding_mask):
        # print('************train_step*************')
        with tf.GradientTape() as tape:
            # 逐个预测序列

            final_dists, _, attentions, p_gen, coverages = model(enc_inp,
                                                                 dec_input,
                                                                 extended_enc_input,
                                                                 max_oov_len,
                                                                 enc_pad_mask=enc_pad_mask,
                                                                 use_coverage=params['use_coverage'],
                                                                 prev_coverage=None)

            batch_loss, log_loss, cov_loss = calc_loss(dec_target, final_dists, padding_mask, attentions,
                                                       params['cov_loss_wt'],
                                                       params['use_coverage'])

        # print('************variables*************')
        variables = model.encoder.trainable_variables + model.decoder.trainable_variables + \
                    model.attention.trainable_variables + model.pointer.trainable_variables

        # print('************gradient*************')
        gradients = tape.gradient(batch_loss, variables)

        # print('************optimizer*************')
        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss, log_loss, cov_loss

    for epoch in range(epochs):
        start = time.time()
        # dataset = batcher(vocab, params)
        total_loss = 0
        step = 0
        for encoder_batch_data, decoder_batch_data in dataset:
            batch_loss, log_loss, cov_loss = train_step(encoder_batch_data["enc_input"],
                                                        encoder_batch_data["extended_enc_input"],
                                                        encoder_batch_data["max_oov_len"],
                                                        decoder_batch_data["dec_input"],
                                                        decoder_batch_data["dec_target"],
                                                        enc_pad_mask=encoder_batch_data["encoder_pad_mask"],
                                                        padding_mask=decoder_batch_data["decoder_pad_mask"])

            total_loss += batch_loss

            step += 1

            if step % 1 == 0:
                if params['use_coverage']:

                    print('Epoch {} Batch {} batch_loss {:.4f} log_loss {:.4f} cov_loss {:.4f}'.format(epoch + 1,
                                                                                                       step,
                                                                                                       batch_loss.numpy(),
                                                                                                       log_loss.numpy(),
                                                                                                       cov_loss.numpy()))
                else:
                    print('Epoch {} Batch {} batch_loss {:.4f}'.format(epoch + 1,
                                                                       step,
                                                                       batch_loss.numpy()))

                if step > params['max_train_steps']:
                    break
        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 1 == 0:
            ckpt_save_path = checkpoint_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))
            print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / step))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
