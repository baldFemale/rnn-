import tensorflow as tf
import numpy as np
import pickle
###########RNN实践到此结束，主要拿着个熟悉RNN,LSTM,tensorflow的保存、读取（因为没做好这一部分所以不能进行中断的训练）
###########因为训练集比较小，也只循环了100次，所以结果不是很好，只比瞎排好一点，但还算满意吧


num_epochs = 100
batch_size = 256
rnn_size = 512
embed_dim = 512
seq_length = 30
learning_rate = 0.004
show_every_n_batch = 30
save_path = 'D:/tensorflow/model.ckpt'


def load():
    int2vocab_dir = 'C:/Users\Administrator\Desktop\RNN/vocab2int.pk'
    int_text_dir = 'C:/Users\Administrator\Desktop\RNN/text.pk'
    pk_text = open(int_text_dir,'rb')
    int_text = pickle.load(pk_text)
    pk_text.close()
    pk_dic = open(int2vocab_dir,'rb')
    int2vocab = pickle.load(pk_dic)
    pk_dic.close()
    # print(int_text,int2vocab)
    for i in range(len(int_text)):
        int_text[i] = int2vocab[int_text[i]]
    return  int_text[:50000], int2vocab

int_text,int2vocab = load()


def load2():
    int2vocab_dir = 'C:/Users\Administrator\Desktop\RNN/int2vocab.pk'
    pk_dic = open(int2vocab_dir,'rb')
    vocab2int = pickle.load(pk_dic)
    pk_dic.close()
    print(vocab2int)
    return vocab2int

vocab2int = load2()


def get_inputs():
    inputs = tf.placeholder(tf.int32,[None,None],name='inputs')
    targets = tf.placeholder(tf.int32,[None,None],name='targets')
    learning_rate = tf.placeholder(tf.float32,name='learning_rate')
    return inputs,targets,learning_rate


def get_embed(input_data,vocab_size,embed_dim):
    embedding = tf.Variable(tf.random_uniform((vocab_size,embed_dim)),dtype=tf.float32)
    return tf.nn.embedding_lookup(embedding,input_data)

#这个batch_size到底是什么？
#batch应该是为了实现小批量梯度下降

def get_init_cell(batch_size,run_size):
    num_layer = 2
    keep_prob = 0.8
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(run_size)
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,output_keep_prob=keep_prob)
    lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*num_layer)
    init_state = lstm_cell.zero_state(batch_size,tf.float32)
    init_state = tf.identity(init_state,'init_state')

    return lstm_cell,init_state


def bulid_rnn(cell,inputs):
    outputs,final_state = tf.nn.dynamic_rnn(cell,inputs,dtype=tf.float32)
    final_state = tf.identity(final_state,'final_state')
    return outputs,final_state


def build_nn(cell,rnn_size,input_data,vocab_size,embed_dim):
    embed = get_embed(input_data,vocab_size,embed_dim)
    outputs,final_state = bulid_rnn(cell,embed)
    logits = tf.contrib.layers.fully_connected(outputs, vocab_size, activation_fn=None,
                                              weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                              biases_initializer=tf.zeros_initializer())
    #本质目的是为了生成一个全连接层，但我不知道能不能实现

    return logits,final_state


# 这一步是因为担心训练集过大，所以决定分成几个batch来训练
def get_batches(int_text,batch_size,seq_length):
    n_batches = (len(int_text)//(batch_size*seq_length))
    # 字符集整除batch大小*每个batch中的字符个数（其实就是每次放进网络的字符串长度）
    batch_origin = np.array(int_text[:n_batches*batch_size*seq_length])
    batch_shifted = np.array(int_text[1:n_batches*batch_size*seq_length+1])
    batch_shifted[-1] = batch_origin[0]
    # 为什么要用一个循环，目测是为了让出现字符都在这个集里
    batch_origin_reshape = np.split(batch_origin.reshape(batch_size,-1),n_batches,1)
    batch_shifted_reshape = np.split(batch_shifted.reshape(batch_size,-1),n_batches,1)

    batches = np.array(list(zip(batch_origin_reshape,batch_shifted_reshape)))
    return batches
# 我的理解这是生成训练集，和word2vec里差不多


def get_tensors(loaded_graph):
    inputs = loaded_graph.get_tensor_by_name("inputs:0")
    initial_state = loaded_graph.get_tensor_by_name("init_state:0")
    final_state = loaded_graph.get_tensor_by_name("final_state:0")
    probs = loaded_graph.get_tensor_by_name("probs:0")
    learning_rate = loaded_graph.get_tensor_by_name("learning_rate:0")
    targets = loaded_graph.get_tensor_by_name("targets:0")
    return inputs, initial_state, final_state, probs,learning_rate,targets
###########定义完成，创建RNN模型####################
from tensorflow.contrib import seq2seq


train_graph = tf.Graph()
with train_graph.as_default():
    vocab_size = len(int2vocab)

    input_text,targets,lr = get_inputs()
    input_data_shape = tf.shape(input_text)

    cell,initial_state = get_init_cell(input_data_shape[0],rnn_size)
    logits,final_state = build_nn(cell,rnn_size,input_text,vocab_size,embed_dim)
    probs = tf.nn.softmax(logits,name='probs')

    loss = seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_data_shape[0],input_data_shape[1]])
    )
    #为什么是这个损失函数，需要了解

    optimizer = tf.train.AdamOptimizer(lr)
    gradients = optimizer.compute_gradients(loss)
    #计算梯度，返回的是一个var-loss的tuple
    capped_gradients = [(tf.clip_by_value(grad,-1,1),var) for grad, var in gradients if grad is not None]
    #clip可以把值域控制在一定范围内
    train_op = optimizer.apply_gradients(capped_gradients)
    #在把梯度按照自己需求处理后，得到更新状态


#####################开始训练#####################
#####################现在我在尝试进行断点的训练###
batches = get_batches(int_text,batch_size,seq_length)
print(batches[0][0])
graph_loaded = tf.Graph()
with tf.Session(graph=graph_loaded) as sess:
    # sess.run(tf.global_variables_initializer())
    loader = tf.train.import_meta_graph(save_path+'.meta')
    loader.restore(sess,save_path)
    input_text,initial_state,final_state,probs,lr,targets = get_tensors(graph_loaded)
    #####################################################################################
    vocab_size = len(int2vocab)
    input_data_shape = tf.shape(input_text)
    cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
    logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size, embed_dim)
    probs = tf.nn.softmax(logits, name='probs')
    loss = seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_data_shape[0], input_data_shape[1]])
    )
    # 为什么是这个损失函数，需要了解
    optimizer = tf.train.AdamOptimizer(lr)
    gradients = optimizer.compute_gradients(loss)
    # 计算梯度，返回的是一个var-loss的tuple
    capped_gradients = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in gradients if grad is not None]
    # clip可以把值域控制在一定范围内
    train_op = optimizer.apply_gradients(capped_gradients)
    ####################################################################################
    for epoch_i in range(num_epochs):
        state = sess.run(initial_state,{input_text:batches[0][0]})
        #需要理解batches到底是什么

        for batch_i,(x,y) in enumerate(batches):
            feed = {
                input_text:x,
                targets:y,
                initial_state:state,
                lr:learning_rate
            }
            train_loss ,state,_ = sess.run([loss,final_state,train_op],feed)

            if (epoch_i*len(batches)+batch_i)%show_every_n_batch==0:
                print(
                    epoch_i,
                    batch_i,
                    len(batches),
                    train_loss
                )

    saver = tf.train.Saver()
    saver.save(sess,save_path)
    print('Model Trained and Saved')

############模型在这里训练结束，接下来是生产文本部分######################
############选词部分####################################
# def pick_word(probabilities,int2vocab):
#     chance = []
#     for idx,prob in enumerate(probabilities):
#         if prob>=0.02:
#             chance.append(int2vocab[idx])
#     print(len(chance))
#     rand = np.random.randint(0,len(chance))
#     return str(chance[rand])
#
# ##########载入模型部分################################
# def get_tensors(loaded_graph):
#     inputs = loaded_graph.get_tensor_by_name("inputs:0")
#     initial_state = loaded_graph.get_tensor_by_name("init_state:0")
#     final_state = loaded_graph.get_tensor_by_name("final_state:0")
#     probs = loaded_graph.get_tensor_by_name("probs:0")
#     return inputs, initial_state, final_state, probs
#
#
# gen_length = 1000
# prime_word = '青'
#
# loaded_graph = tf.Graph()
# with tf.Session(graph=loaded_graph) as sess:
#     loader = tf.train.import_meta_graph(save_path+'.meta')
#     loader.restore(sess,save_path)
#     input_text,initial_state,final_state,probs = get_tensors(loaded_graph)
#
#
#     gen_sentence = [prime_word]
#     prev_state = sess.run(initial_state,{input_text:np.array([[1]])})
#
#     s = '青'
#     for n in range(gen_length):
#         dyn_input = [[int2vocab[word] for word in gen_sentence[-seq_length:]]]
#         dyn_seq_len = len(dyn_input[0])
#         probcabilities,prev_final_state = sess.run(
#             [probs,final_state],
#             {input_text:dyn_input,initial_state:prev_state}
#         )
#         pred_word = pick_word(probabilities[0][dyn_seq_len-1],vocab2int)
#         s = s+str(pred_word)
#         gen_sentence.append(pred_word)
# print(s)