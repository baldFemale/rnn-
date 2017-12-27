import tensorflow as tf
import numpy as np
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq
import time


#设定一些初始的参数、数据集
num_encoder_symbols = 10
num_decoder_symbols = 16
input_seq_len = 5
output_seq_len = input_seq_len
train_set = [[[1, 3, 5], [7, 9, 11]], [[3, 5, 7], [9, 11, 13]]]
PAD = 0
GO = 1
size = 8
learning_rate = 0.1


def get_sample():
    encoder_input_0 = [PAD]*(input_seq_len-len(train_set[0][0]))+train_set[0][0]
    encoder_input_1 = [PAD]*(input_seq_len-len(train_set[1][0]))+train_set[1][0]
    decoder_input_0 = [GO]+train_set[0][1]+[PAD]*(output_seq_len-len(train_set[0][1])-1)
    decoder_input_1 = [GO]+train_set[1][1]+[PAD]*(output_seq_len-len(train_set[1][1])-1)
    #tensorflow的接口中对tensor的要求依次是：句子长度、batch_size
    encoder_input = []
    for i in range(input_seq_len):
        encoder_input.append(
        np.array([encoder_input_0[i],encoder_input_1[i]],dtype=np.int32)
        )
    decoder_input = []
    for i in range(output_seq_len):
        decoder_input.append(
            np.array([decoder_input_0[i],decoder_input_1[i]],dtype=np.int32)
        )
    targets_weighs = []
    for i in range(output_seq_len):
        targets_weighs.append(
            np.array(
                [0.0 if i==output_seq_len-1 or decoder_input_0[i]==PAD else 1.0,
                0.0 if i==output_seq_len-1 or decoder_input_1[i]==PAD else 1.0],
                dtype=np.float32
            )
        )
    return encoder_input,decoder_input,targets_weighs


def get_model():
    # 这个方法需要的参数分别是：inputs_tensor,decoder_tensor,cell,类似与vocab_size的symbols,虽然我不知道encoder_symbolsy有什么用
    # 然后是embed_size，应该和cell的size一样，然后是需不需要softmax，decode_inputs是来自前面的RNNcell还是我们自己输入，最后是数
    # 据类型
    '''
    embedding_attention_seq2seq(
        encoder_inputs,
        decoder_inputs,
        cell,
        num_encoder_symbols,
        num_decoder_symbols,
        embedding_size,
        num_heads=1,
        output_projection=None,
        feed_previous=False,
        dtype=None,
        scope=None,
        initial_state_attention=False
    )
    '''
    encoder_inputs = []
    decoder_inputs = []
    targets_weigh = []
    for i in range(input_seq_len):
        encoder_inputs.append(tf.placeholder(shape=[None],dtype=tf.int32,name="encoder{0}".format(i)))
    for i in range(output_seq_len):
        decoder_inputs.append(tf.placeholder(shape=[None],dtype=tf.int32,name="decode{0}".format(i)))
    for i in range(output_seq_len):
        targets_weigh.append(
            tf.placeholder(shape=[None],dtype=tf.float32,name="weight{0}".format(i))
        )
    targets = [decoder_inputs[i] for i in range(1,output_seq_len)]
    targets.append(np.zeros(shape=[2],dtype=np.int32))
    cell = tf.nn.rnn_cell.BasicLSTMCell(size)
    outputs,_ = seq2seq.embedding_attention_seq2seq(
        encoder_inputs,
        decoder_inputs,
        cell,
        num_encoder_symbols=num_encoder_symbols,
        num_decoder_symbols=num_decoder_symbols,
        embedding_size=size,
        output_projection=None,
        feed_previous=False,
        dtype=tf.float32
    )
    loss = seq2seq.sequence_loss(
        outputs,targets,targets_weigh
    )
    opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    update = opt.apply_gradients(opt.compute_gradients(loss))
    saver = tf.train.Saver(tf.global_variables())
    return encoder_inputs,decoder_inputs,targets_weigh,outputs,loss,update,saver
    pass

'''
sequence_loss(
    logits,           5*2*16,长度*batch*symbols,也就是我们的outputs
    targets,          5*2,长度*batch
    weights,          和target类型一样,加权交叉熵损失函数的权重矩阵
    average_across_timesteps=True,
    average_across_batch=True,
    softmax_loss_function=None,
    name=None
)
)
sequence2sequence的loss function
'''


def main():
    with tf.Session() as sess:
        sample_encoder_inputs,sample_decoder_inputs,sample_targets_weight = get_sample()
        encoder_inputs,decoder_inputs,targets_weigh,outputs,loss,update,saver = get_model()
        feed = {}
        for i in range(input_seq_len):
            feed[encoder_inputs[i].name] = sample_encoder_inputs[i]
        for i in range(output_seq_len):
            feed[decoder_inputs[i].name] = sample_decoder_inputs[i]
        for i in range(output_seq_len):
            feed[targets_weigh[i].name] = sample_targets_weight[i]
        sess.run(tf.global_variables_initializer())
        for step in range(200):
            [loss_ret,_ ]= sess.run([loss,update],feed_dict=feed)
            print(loss_ret)
            if step % 10 == 0:
                print('step=', step, 'loss=', loss_ret)
            # 模型持久化
            saver.save(sess, 'D:/tensorflow/./model/demo')


if __name__=='__main__':
    main()