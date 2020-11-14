# coding:utf-8
import argparse
import shutil
from util import *
import time
import numpy as np
import sys
import random
from data_iterator import DataIterator
from model import *
from tensorboardX import SummaryWriter
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('-p', type=str, default='train', help='train | test')
parser.add_argument('--dataset', type=str, default='ml-1m', help='ml-1m | ml-10m | book | taobao')
parser.add_argument('--random_seed', type=int, default=19)
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--num_interest', type=int, default=4)
parser.add_argument('--model_type', type=str, default='none', help='DNN | GRU4REC | ..')
parser.add_argument('--learning_rate', type=float, default=0.001, help='')
parser.add_argument('--max_iter', type=int, default=1000, help='(k)')
parser.add_argument('--patience', type=int, default=50)
parser.add_argument('--coef', default=None)
parser.add_argument('--topN', type=int, default=50)
parser.add_argument('--test_iter', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--maxlen', type=int, default=50)
parser.add_argument('--prefix', type=str, default='../data/')

best_metric = 0


def get_model(dataset, model_type, item_count, batch_size, maxlen):
    if model_type == 'DNN':
        model = Model_DNN(item_count, args.embedding_dim, args.hidden_size, batch_size, maxlen)
    elif model_type == 'MIND':
        relu_layer = True if dataset == 'book' else False
        model = Model_MIND(item_count, args.embedding_dim, args.hidden_size, batch_size, args.num_interest, maxlen,
                           relu_layer=relu_layer)
    elif model_type == 'ComiRec-DR':
        model = Model_ComiRec_DR(item_count, args.embedding_dim, args.hidden_size, batch_size, args.num_interest,
                                 maxlen)
    elif model_type == 'ComiRec-SA':
        model = Model_ComiRec_SA(item_count, args.embedding_dim, args.hidden_size, batch_size, args.num_interest,
                                 maxlen)
    else:
        print("Invalid model_type : %s", model_type)
        return
    return model


def get_exp_name(dataset, model_type, batch_size, lr, maxlen, save=True):
    extr_name = input('Please input the experiment name: ')
    para_name = '_'.join([dataset, model_type, 'b' + str(batch_size), 'lr' + str(lr), 'd' + str(args.embedding_dim),
                          'len' + str(maxlen)])
    exp_name = para_name + '_' + extr_name

    while os.path.exists('runs/' + exp_name) and save:
        flag = input('The exp name already exists. Do you want to cover? (y/n)')
        if flag == 'y' or flag == 'Y':
            shutil.rmtree('runs/' + exp_name)
            break
        else:
            extr_name = input('Please input the experiment name: ')
            exp_name = para_name + '_' + extr_name

    return exp_name


def train(
        train_file,
        valid_file,
        test_file,
        cate_file,
        item_count,
        dataset="book",
        batch_size=128,
        maxlen=100,
        test_iter=50,
        model_type='DNN',
        lr=0.001,
        max_iter=100,
        patience=20
):
    exp_name = get_exp_name(dataset, model_type, batch_size, lr, maxlen)
    best_model_path = "best_model/" + exp_name + '/'
    gpu_options = tf.GPUOptions(allow_growth=True)
    writer = SummaryWriter('runs/' + exp_name)

    item_cate_map = load_item_cate(cate_file)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # (user_id_list, item_id_list), (hist_item_list, hist_mask_list)
        train_data = DataIterator(train_file, batch_size, maxlen, train_flag=0)
        valid_data = DataIterator(valid_file, batch_size, maxlen, train_flag=1)

        model = get_model(dataset, model_type, item_count, batch_size, maxlen)

        # 初始化参数
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        print('training begin')
        sys.stdout.flush()

        start_time = time.time()
        iter = 0
        try:
            loss_sum = 0.0
            trials = 0

            # train_data: userid, itemid, sql_num
            for src, tgt in train_data:
                data_iter = prepare_data(src, tgt)
                loss = model.train(sess, list(data_iter) + [lr])

                loss_sum += loss
                iter += 1

                if iter % test_iter == 0:
                    metrics = evaluate_full(sess, args.topN, args.hidden_size, valid_data, model, best_model_path,
                                            batch_size, item_cate_map)
                    log_str = 'iter: %d, train loss: %.4f' % (iter, loss_sum / test_iter)
                    if metrics != {}:
                        log_str += ', ' + ', '.join(
                            ['valid ' + key + ': %.6f' % value for key, value in metrics.items()])
                    print(exp_name)
                    print(log_str)

                    writer.add_scalar('train/loss', loss_sum / test_iter, iter)
                    if metrics != {}:
                        for key, value in metrics.items():
                            writer.add_scalar('eval/' + key, value, iter)

                    if 'recall' in metrics:
                        recall = metrics['recall']
                        global best_metric
                        if recall > best_metric:
                            best_metric = recall
                            model.save(sess, best_model_path)
                            trials = 0
                        else:
                            trials += 1
                            if trials > patience:
                                break

                    loss_sum = 0.0
                    test_time = time.time()
                    print("time interval: %.4f min" % ((test_time - start_time) / 60.0))
                    sys.stdout.flush()

                if iter >= max_iter * 1000:
                    break
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

        model.restore(sess, best_model_path)

        metrics = evaluate_full(sess, args.topN, args.hidden_size, valid_data, model, best_model_path, batch_size,
                                item_cate_map, save=False)
        print(', '.join(['valid ' + key + ': %.6f' % value for key, value in metrics.items()]))

        test_data = DataIterator(test_file, batch_size, maxlen, train_flag=2)
        metrics = evaluate_full(sess, args.topN, args.hidden_size, test_data, model, best_model_path, batch_size,
                                item_cate_map, save=False)
        print(', '.join(['test ' + key + ': %.6f' % value for key, value in metrics.items()]))


def test(
        test_file,
        cate_file,
        item_count,
        dataset="book",
        batch_size=128,
        maxlen=100,
        model_type='DNN',
        lr=0.001
):
    exp_name = get_exp_name(dataset, model_type, batch_size, lr, maxlen, save=False)
    best_model_path = "best_model/" + exp_name + '/'
    gpu_options = tf.GPUOptions(allow_growth=True)
    model = get_model(dataset, model_type, item_count, batch_size, maxlen)
    item_cate_map = load_item_cate(cate_file)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model.restore(sess, best_model_path)

        test_data = DataIterator(test_file, batch_size, maxlen, train_flag=2)
        metrics = evaluate_full(sess, args.hidden_units, args.topN, test_data, model, best_model_path, batch_size,
                                item_cate_map, save=False, coef=args.coef)
        print(', '.join(['test ' + key + ': %.6f' % value for key, value in metrics.items()]))


def output(
        item_count,
        dataset="book",
        batch_size=128,
        maxlen=100,
        model_type='DNN',
        lr=0.001
):
    exp_name = get_exp_name(dataset, model_type, batch_size, lr, maxlen, save=False)
    best_model_path = "best_model/" + exp_name + '/'
    gpu_options = tf.GPUOptions(allow_growth=True)
    model = get_model(dataset, model_type, item_count, batch_size, maxlen)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model.restore(sess, best_model_path)
        item_embs = model.output_item(sess)
        np.save('output/' + exp_name + '_emb.npy', item_embs)


if __name__ == '__main__':
    print(sys.argv)
    args = parser.parse_args()
    SEED = args.random_seed

    tf.set_random_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    train_name = 'train'
    valid_name = 'valid'
    test_name = 'test'

    if args.dataset == 'ml-1m':
        # path = '/content/MyComiRec/data/ml-1m_data/'
        path = args.prefix + 'ml-1m_data/'
        item_count = 3417
        batch_size = args.batch_size
        maxlen = args.maxlen
        test_iter = args.test_iter
    elif args.dataset == 'ml-10m':
        # path = '/content/MyComiRec/data/ml-10m_data/'
        path = args.prefix + 'ml-10m_data/'
        item_count = 10197
        batch_size = args.batch_size
        maxlen = args.maxlen
        test_iter = args.test_iter
    elif args.dataset == 'taobao':
        path = args.prefix + 'taobao_data/'
        item_count = 1708531
        batch_size = 256
        maxlen = 50
        test_iter = 500
    elif args.dataset == 'book':
        path = args.prefix + 'book_data/'
        item_count = 367983
        batch_size = 128
        maxlen = 20
        test_iter = 1000

    train_file = path + args.dataset + '_train.txt'
    valid_file = path + args.dataset + '_valid.txt'
    test_file = path + args.dataset + '_test.txt'
    cate_file = path + args.dataset + '_item_cate.txt'
    dataset = args.dataset

    if args.p == 'train':
        train(train_file=train_file, valid_file=valid_file, test_file=test_file, cate_file=cate_file,
              item_count=item_count, dataset=dataset, batch_size=batch_size, maxlen=maxlen, test_iter=test_iter,
              model_type=args.model_type, lr=args.learning_rate, max_iter=args.max_iter, patience=args.patience)
    elif args.p == 'test':
        test(test_file=test_file, cate_file=cate_file, item_count=item_count, dataset=dataset, batch_size=batch_size,
             maxlen=maxlen, model_type=args.model_type, lr=args.learning_rate)
    elif args.p == 'output':
        output(item_count=item_count, dataset=dataset, batch_size=batch_size, maxlen=maxlen,
               model_type=args.model_type, lr=args.learning_rate)
    else:
        print('do nothing...')