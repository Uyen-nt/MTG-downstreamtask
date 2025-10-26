#################################################################
# Code written by Edward Choi (mp2893@gatech.edu)
# Ported to modern Python/NumPy by Grok (xAI)
# - Chỉ sửa cú pháp, không thay đổi logic
#################################################################

import sys
import random
import time
import argparse
from collections import OrderedDict
import pickle
import numpy as np
import os

import theano
import theano.tensor as T
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# Cấu hình Theano
config.floatX = 'float32'

_TEST_RATIO = 0.2
_VALIDATION_RATIO = 0.1


def unzip(zipped):
    new_params = OrderedDict()
    for key, value in zipped.items():
        new_params[key] = value.get_value()
    return new_params


def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)


def get_random_weight(dim1, dim2, left=-0.1, right=0.1):
    return np.random.uniform(left, right, (dim1, dim2)).astype(config.floatX)


def load_embedding(options):
    m = np.load(options['embFile'])
    w = (m['w'] + m['w_tilde']) / 2.0
    return w


def init_params(options):
    params = OrderedDict()

    np.random.seed(0)
    inputDimSize = options['inputDimSize']
    numAncestors = options['numAncestors']
    embDimSize = options['embDimSize']
    hiddenDimSize = options['hiddenDimSize']
    attentionDimSize = options['attentionDimSize']
    numClass = options['numClass']

    params['W_emb'] = get_random_weight(inputDimSize + numAncestors, embDimSize)
    if len(options['embFile']) > 0:
        params['W_emb'] = load_embedding(options)
        options['embDimSize'] = params['W_emb'].shape[1]
        embDimSize = options['embDimSize']

    params['W_attention'] = get_random_weight(embDimSize * 2, attentionDimSize)
    params['b_attention'] = np.zeros(attentionDimSize, dtype=config.floatX)
    params['v_attention'] = np.random.uniform(-0.1, 0.1, attentionDimSize).astype(config.floatX)

    params['W_gru'] = get_random_weight(embDimSize, 3 * hiddenDimSize)
    params['U_gru'] = get_random_weight(hiddenDimSize, 3 * hiddenDimSize)
    params['b_gru'] = np.zeros(3 * hiddenDimSize, dtype=config.floatX)

    params['W_output'] = get_random_weight(hiddenDimSize, numClass)
    params['b_output'] = np.zeros(numClass, dtype=config.floatX)

    return params


def init_tparams(params):
    tparams = OrderedDict()
    for key, value in params.items():
        tparams[key] = theano.shared(value, name=key)
    return tparams


def dropout_layer(state_before, use_noise, trng, prob):
    proj = T.switch(
        use_noise,
        (state_before * trng.binomial(state_before.shape, p=prob, n=1, dtype=state_before.dtype)),
        state_before * 0.5
    )
    return proj


def _slice(_x, n, dim):
    if _x.ndim == 3:
        return _x[:, :, n * dim:(n + 1) * dim]
    return _x[:, n * dim:(n + 1) * dim]


def gru_layer(tparams, emb, options):
    hiddenDimSize = options['hiddenDimSize']
    timesteps = emb.shape[0]
    n_samples = emb.shape[1] if emb.ndim == 3 else 1

    def stepFn(wx, h, U_gru):
        uh = T.dot(h, U_gru)
        r = T.nnet.sigmoid(_slice(wx, 0, hiddenDimSize) + _slice(uh, 0, hiddenDimSize))
        z = T.nnet.sigmoid(_slice(wx, 1, hiddenDimSize) + _slice(uh, 1, hiddenDimSize))
        h_tilde = T.tanh(_slice(wx, 2, hiddenDimSize) + r * _slice(uh, 2, hiddenDimSize))
        h_new = z * h + ((1.0 - z) * h_tilde)
        return h_new

    Wx = T.dot(emb, tparams['W_gru']) + tparams['b_gru']
    results, updates = theano.scan(
        fn=stepFn,
        sequences=[Wx],
        outputs_info=T.alloc(numpy_floatX(0.0), n_samples, hiddenDimSize),
        non_sequences=[tparams['U_gru']],
        name='gru_layer',
        n_steps=timesteps
    )
    return results


def generate_attention(tparams, leaves, ancestors):
    attentionInput = T.concatenate([tparams['W_emb'][leaves], tparams['W_emb'][ancestors]], axis=2)
    mlpOutput = T.tanh(T.dot(attentionInput, tparams['W_attention']) + tparams['b_attention'])
    preAttention = T.dot(mlpOutput, tparams['v_attention'])
    attention = T.nnet.softmax(preAttention)
    return attention


def softmax_layer(tparams, emb):
    nom = T.exp(T.dot(emb, tparams['W_output']) + tparams['b_output'])
    denom = nom.sum(axis=2, keepdims=True)
    output = nom / denom
    return output


def build_model(tparams, leavesList, ancestorsList, options):
    dropoutRate = options['dropoutRate']
    trng = RandomStreams(123)
    use_noise = theano.shared(numpy_floatX(0.0))

    x = T.tensor3('x', dtype=config.floatX)
    y = T.tensor3('y', dtype=config.floatX)
    mask = T.matrix('mask', dtype=config.floatX)
    lengths = T.vector('lengths', dtype=config.floatX)

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    embList = []
    for leaves, ancestors in zip(leavesList, ancestorsList):
        tempAttention = generate_attention(tparams, leaves, ancestors)
        tempEmb = (tparams['W_emb'][ancestors] * tempAttention[:, :, None]).sum(axis=1)
        embList.append(tempEmb)

    emb = T.concatenate(embList, axis=0)
    x_emb = T.tanh(T.dot(x, emb))
    hidden = gru_layer(tparams, x_emb, options)
    hidden = dropout_layer(hidden, use_noise, trng, dropoutRate)
    y_hat = softmax_layer(tparams, hidden) * mask[:, :, None]

    logEps = 1e-8
    cross_entropy = -(y * T.log(y_hat + logEps) + (1.0 - y) * T.log(1.0 - y_hat + logEps))
    output_loglikelihood = cross_entropy.sum(axis=2).sum(axis=0) / lengths
    cost_noreg = T.mean(output_loglikelihood)

    if options['L2'] > 0.0:
        cost = cost_noreg + options['L2'] * (
            (tparams['W_output'] ** 2).sum() +
            (tparams['W_attention'] ** 2).sum() +
            (tparams['v_attention'] ** 2).sum()
        )
    else:
        cost = cost_noreg

    return use_noise, x, y, mask, lengths, cost, cost_noreg, y_hat


def load_data(seqFile, labelFile, timeFile=''):
    sequences = np.array(pickle.load(open(seqFile, 'rb')))
    labels = np.array(pickle.load(open(labelFile, 'rb')))
    times = None
    if timeFile:
        times = np.array(pickle.load(open(timeFile, 'rb')))

    np.random.seed(0)
    dataSize = len(labels)
    ind = np.random.permutation(dataSize)
    nTest = int(_TEST_RATIO * dataSize)
    nValid = int(_VALIDATION_RATIO * dataSize)

    test_indices = ind[:nTest]
    valid_indices = ind[nTest:nTest + nValid]
    train_indices = ind[nTest + nValid:]

    train_set_x = sequences[train_indices]
    train_set_y = labels[train_indices]
    test_set_x = sequences[test_indices]
    test_set_y = labels[test_indices]
    valid_set_x = sequences[valid_indices]
    valid_set_y = labels[valid_indices]

    train_set_t = test_set_t = valid_set_t = None
    if times is not None:
        train_set_t = times[train_indices]
        test_set_t = times[test_indices]
        valid_set_t = times[valid_indices]

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    train_sorted_index = len_argsort(train_set_x)
    train_set_x = [train_set_x[i] for i in train_sorted_index]
    train_set_y = [train_set_y[i] for i in train_sorted_index]

    valid_sorted_index = len_argsort(valid_set_x)
    valid_set_x = [valid_set_x[i] for i in valid_sorted_index]
    valid_set_y = [valid_set_y[i] for i in valid_sorted_index]

    test_sorted_index = len_argsort(test_set_x)
    test_set_x = [test_set_x[i] for i in test_sorted_index]
    test_set_y = [test_set_y[i] for i in test_sorted_index]

    if times is not None:
        train_set_t = [train_set_t[i] for i in train_sorted_index]
        valid_set_t = [valid_set_t[i] for i in valid_sorted_index]
        test_set_t = [test_set_t[i] for i in test_sorted_index]

    return (train_set_x, train_set_y, train_set_t), \
           (valid_set_x, valid_set_y, valid_set_t), \
           (test_set_x, test_set_y, test_set_t)


def adadelta(tparams, grads, x, y, mask, lengths, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.0), name=f'{k}_grad') for k, p in tparams.items()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.0), name=f'{k}_rup2') for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.0), name=f'{k}_rgrad2') for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(
        [x, y, mask, lengths], cost,
        updates=zgup + rg2up,
        name='adadelta_f_grad_shared'
    )

    updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2)) for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(list(tparams.values()), updir)]

    f_update = theano.function(
        [], [], updates=ru2up + param_up,
        on_unused_input='ignore',
        name='adadelta_f_update'
    )

    return f_grad_shared, f_update


def padMatrix(seqs, labels, options):
    lengths = np.array([len(seq) - 1 for seq in seqs])
    n_samples = len(seqs)
    maxlen = np.max(lengths)

    x = np.zeros((maxlen, n_samples, options['inputDimSize']), dtype=config.floatX)
    y = np.zeros((maxlen, n_samples, options['numClass']), dtype=config.floatX)
    mask = np.zeros((maxlen, n_samples), dtype=config.floatX)

    for idx, (seq, lseq) in enumerate(zip(seqs, labels)):
        for xvec, subseq in zip(x[:, idx, :], seq[:-1]):
            xvec[subseq] = 1.0
        for yvec, subseq in zip(y[:, idx, :], lseq[1:]):
            yvec[subseq] = 1.0
        mask[:lengths[idx], idx] = 1.0

    lengths = lengths.astype(config.floatX)
    return x, y, mask, lengths


def calculate_cost(test_model, dataset, options):
    batchSize = options['batchSize']
    n_batches = int(np.ceil(len(dataset[0]) / batchSize))
    costSum = 0.0
    dataCount = 0
    for index in range(n_batches):
        batchX = dataset[0][index * batchSize:(index + 1) * batchSize]
        batchY = dataset[1][index * batchSize:(index + 1) * batchSize]
        x, y, mask, lengths = padMatrix(batchX, batchY, options)
        cost = test_model(x, y, mask, lengths)
        costSum += cost * len(batchX)
        dataCount += len(batchX)
    return costSum / dataCount if dataCount > 0 else 0.0


def print2file(buf, outFile):
    with open(outFile, 'a', encoding='utf-8') as f:
        f.write(buf + '\n')


def build_tree(treeFile):
    treeMap = pickle.load(open(treeFile, 'rb'))
    ancestors = np.array(list(treeMap.values()), dtype='int32')
    ancSize = ancestors.shape[1]
    leaves = np.array([[k] * ancSize for k in treeMap.keys()], dtype='int32')
    return leaves, ancestors


def train_GRAM(
    seqFile='seqFile.txt',
    labelFile='labelFile.txt',
    treeFile='tree.txt',
    embFile='',
    outFile='out.txt',
    inputDimSize=100,
    numAncestors=100,
    embDimSize=100,
    hiddenDimSize=200,
    attentionDimSize=200,
    max_epochs=100,
    L2=0.0,
    numClass=26679,
    batchSize=100,
    dropoutRate=0.5,
    logEps=1e-8,
    verbose=False
):
    options = locals().copy()

    leavesList = []
    ancestorsList = []
    for i in range(5, 0, -1):
        level_file = f"{treeFile}.level{i}.pk"
        if not os.path.exists(level_file):
            continue
        leaves, ancestors = build_tree(level_file)
        sharedLeaves = theano.shared(leaves, name=f'leaves{i}')
        sharedAncestors = theano.shared(ancestors, name=f'ancestors{i}')
        leavesList.append(sharedLeaves)
        ancestorsList.append(sharedAncestors)

    if not leavesList:
        raise FileNotFoundError(f"Không tìm thấy file .level*.pk nào cho prefix: {treeFile}")

    print('Building the model ... ', end='', flush=True)
    params = init_params(options)
    tparams = init_tparams(params)
    use_noise, x, y, mask, lengths, cost, cost_noreg, y_hat = build_model(tparams, leavesList, ancestorsList, options)
    get_cost = theano.function([x, y, mask, lengths], cost_noreg, name='get_cost')
    print('done!!')

    print('Constructing the optimizer ... ', end='', flush=True)
    grads = T.grad(cost, wrt=list(tparams.values()))
    f_grad_shared, f_update = adadelta(tparams, grads, x, y, mask, lengths, cost)
    print('done!!')

    print('Loading data ... ', end='', flush=True)
    trainSet, validSet, testSet = load_data(seqFile, labelFile)
    n_batches = int(np.ceil(len(trainSet[0]) / batchSize))
    print('done!!')

    print('Optimization start !!')
    bestValidCost = float('inf')
    bestTestCost = 0.0
    bestTrainCost = 0.0
    bestEpoch = 0
    logFile = outFile + '.log'

    for epoch in range(max_epochs):
        costVec = []
        startTime = time.time()

        for index in random.sample(range(n_batches), n_batches):
            use_noise.set_value(1.0)
            batchX = trainSet[0][index * batchSize:(index + 1) * batchSize]
            batchY = trainSet[1][index * batchSize:(index + 1) * batchSize]
            x, y, mask, lengths = padMatrix(batchX, batchY, options)
            costValue = f_grad_shared(x, y, mask, lengths)
            f_update()
            costVec.append(costValue)

        duration = time.time() - startTime
        use_noise.set_value(0.0)
        trainCost = np.mean(costVec)
        validCost = calculate_cost(get_cost, validSet, options)
        testCost = calculate_cost(get_cost, testSet, options)

        buf = f'Epoch:{epoch}, Duration:{duration:.2f}, Train_Cost:{trainCost:.6f}, Valid_Cost:{validCost:.6f}, Test_Cost:{testCost:.6f}'
        print(buf)
        print2file(buf, logFile)

        if validCost < bestValidCost:
            bestValidCost = validCost
            bestTestCost = testCost
            bestTrainCost = trainCost
            bestEpoch = epoch
            tempParams = unzip(tparams)
            np.savez_compressed(f"{outFile}.{epoch}", **tempParams)

    buf = f'Best Epoch:{bestEpoch}, Train_Cost:{bestTrainCost:.6f}, Valid_Cost:{bestValidCost:.6f}, Test_Cost:{bestTestCost:.6f}'
    print(buf)
    print2file(buf, logFile)


def parse_arguments(parser):
    parser.add_argument('seq_file', type=str, help='Path to visit sequences')
    parser.add_argument('label_file', type=str, help='Path to labels')
    parser.add_argument('tree_file', type=str, help='Prefix of tree files (e.g., tree_mimic3)')
    parser.add_argument('out_file', type=str, help='Output model path')
    parser.add_argument('--embed_file', type=str, default='', help='Pretrained embedding file')
    parser.add_argument('--embed_size', type=int, default=128)
    parser.add_argument('--rnn_size', type=int, default=128)
    parser.add_argument('--attention_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--L2', type=float, default=0.001)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--log_eps', type=float, default=1e-8)
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


def calculate_dimSize(seqFile):
    seqs = pickle.load(open(seqFile, 'rb'))
    codeSet = set()
    for patient in seqs:
        for visit in patient:
            for code in visit:
                codeSet.add(code)
    return max(codeSet) + 1


def get_rootCode(treeFile):
    tree = pickle.load(open(treeFile, 'rb'))
    return list(tree.values())[0][1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    inputDimSize = calculate_dimSize(args.seq_file)
    numClass = calculate_dimSize(args.label_file)
    numAncestors = get_rootCode(args.tree_file + '.level2.pk') - inputDimSize + 1

    train_GRAM(
        seqFile=args.seq_file,
        labelFile=args.label_file,
        treeFile=args.tree_file,
        outFile=args.out_file,
        inputDimSize=inputDimSize,
        numAncestors=numAncestors,
        numClass=numClass,
        embFile=args.embed_file,
        embDimSize=args.embed_size,
        hiddenDimSize=args.rnn_size,
        attentionDimSize=args.attention_size,
        batchSize=args.batch_size,
        max_epochs=args.n_epochs,
        L2=args.L2,
        dropoutRate=args.dropout_rate,
        logEps=args.log_eps,
        verbose=args.verbose
    )
