import argparse
import sys
from AutoGEL.utils import *
from AutoGEL.log import *
from train import *
# from simulate import *



#sys.path.append('/data4/zhengxin/code/AutoGEL/extra/v1')
def derive_arch(model):
    model.Z_agg_hard = torch.tensor(model.searched_arch_z['agg'], device=model.device)
    model.Z_combine_hard = torch.tensor(model.searched_arch_z['combine'], device=model.device)
    model.Z_act_hard = torch.tensor(model.searched_arch_z['act'], device=model.device)
    model.Z_layer_connect_hard = torch.tensor(model.searched_arch_z['layer_connect'], device=model.device)
    model.Z_layer_agg_hard = torch.tensor(model.searched_arch_z['layer_agg'], device=model.device)
    #model.Z_pool_hard = torch.tensor(model.searched_arch_z['pool'], device=model.device)
    return model
def retrain(model, train_loader, val_loader, test_loader, train_mask, val_mask, test_mask, args, logger):
    device = get_device(args)
    model = derive_arch(model)

    logger.info('Derived z')
    logger.info(model.searched_arch_z)
    logger.info('Derived arch')
    logger.info(model.searched_arch_op)

    def weight_reset(m):
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()
    model.apply(weight_reset)
    model.to(device)

    # train_loader, val_loader, test_loader = dataloaders
    optimizer = get_optimizer(model, args)
    metric = args.metric
    #     recorder = Recorder(metric)
    recorder = RetrainRecorder(metric)
    for step in range(args.retrain_epoch):
        optimize_model(model, train_loader, train_mask, optimizer, device, args)
        train_loss, train_acc, train_auc = eval_model(model, train_loader, train_mask, device, args, split='train')
        val_loss, val_acc, val_auc = eval_model(model, val_loader, val_mask, device, args, split='val')
        test_loss, test_acc, test_auc = eval_model(model, test_loader, test_mask, device, args, split='test')

        recorder.update(train_acc, train_auc, val_acc, val_auc, test_acc, test_auc)
        #         recorder.update(train_acc, train_auc, val_acc, val_auc)
        #recorder.update(train_acc, train_auc, test_acc, test_auc)

        # logger.info('epoch %d best test %s: %.4f, retrain loss: %.4f; retrain %s: %.4f test %s: %.4f' %
        #             (step, metric, recorder.get_best_metric()[0], train_loss,
        #              metric, recorder.get_latest_metrics()[0],
        #              metric, recorder.get_latest_metrics()[1]))
    logger.info('(Retrain Stage) best val acc: %.4f (epoch: %d)' % recorder.get_best_acc_val())
    logger.info('(Retrain Stage) best test acc: %.4f (epoch: %d)' % recorder.get_best_acc())


    results, max_step = recorder.get_best_metric()
    model.max_step = max_step
    model.best_metric_retrain = results
    val_results, val_max_step = recorder.get_best_metric_val()
    ans = {'valid_acc': val_results , 'test_acc': results}
    return model, ans

def main_autogel(searched_arch_op, dataname, seed ,gpu = 0  ):
    parser = argparse.ArgumentParser('Interface for Auto-GNN framework')

    parser.add_argument('--train_ratio', type=float, default=0.6, help='ratio of the train against whole')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio of the val against whole')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='ratio of the test against whole')
    parser.add_argument('--model', type=str, default='Auto-GNN', help='model to use')
    parser.add_argument('--metric', type=str, default='acc', help='metric for evaluating performance', choices=['acc', 'auc'])
    parser.add_argument('--gpu', type=int, default=gpu, help='gpu id')
    parser.add_argument('--directed', type=bool, default=False, help='(Currently unavailable) whether to treat the graph as directed')
    parser.add_argument('--parallel', default=False, action='store_true',
                        help='(Currently unavailable) whether to use multi cpu cores to prepare data')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer to use')

    parser.add_argument('--task', type=str, default='node', help='type of task', choices=['node', 'graph'])
    parser.add_argument('--dataset', type=str, default=dataname, help='dataset name')  # choices=['PROTEINS', 'IMDB-BINARY', 'IMDB-MULTI', 'BZR', 'COX2', 'DD', 'ENZYMES', 'NCI1']
    parser.add_argument('--seed', type=int, default=seed, help='seed to initialize all the random modules')
    parser.add_argument('--clip', type=float, default=0.1, help='gradient clipping')
    # general model and training setting
    parser.add_argument('--epoch', type=int, default=20, help='number of epochs to search')
    parser.add_argument('--retrain_epoch', type=int, default=200, help='number of epochs to retrain')

    parser.add_argument('--layers', type=int, default=2, help='largest number of layers')
    parser.add_argument('--hidden_features', type=int, default=128, help='hidden dimension')
    parser.add_argument('--bs', type=int, default=128, help='minibatch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--l2', type=float, default=1e-4, help='l2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    #parser.add_argument('--dropout', type=float, default=0, help='dropout rate')


    # logging & debug
    parser.add_argument('--log_dir', type=str, default='./log/', help='log directory')

    parser.add_argument('--summary_file', type=str, default='result_summary.log', help='brief summary of training result')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='whether to use debug mode')

    op2z_mapping = {
        'agg': {'sum': [1.0, 0.0, 0.0], 'mean': [0.0, 1.0, 0.0], 'max': [0.0, 0.0, 1.0]},
        'combine': {'sum': [1.0, 0.0], 'concat': [0.0, 1.0]},
        'act': {'relu': [1.0, 0.0], 'prelu': [0.0, 1.0]},
        'layer_connect': {'stack': [1.0, 0.0, 0.0], 'skip_sum': [0.0, 1.0, 0.0], 'skip_cat': [0.0, 0.0, 1.0]},
        'layer_agg': {'none': [1.0, 0.0, 0.0], 'concat': [0.0, 1.0, 0.0], 'max_pooling': [0.0, 0.0, 1.0]},
        'pool': {'global_add_pool': [1.0, 0.0, 0.0], 'global_mean_pool': [0.0, 1.0, 0.0],
                 'global_max_pool': [0.0, 0.0, 1.0]}
    }
    sys_argv = sys.argv
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    check(args)
    logger = set_up_log(args, sys_argv)
    set_random_seed(args)

    if args.dataset != 'PPI':
        dataset, in_features, out_features = get_data(args=args, logger=logger)
        train_mask, val_mask, test_mask = reset_mask(dataset, args=args, logger=logger, stratify=None)
        train_loader, val_loader, test_loader = get_loader(dataset, args=args)
    else:
        in_features, out_features, train_mask, val_mask, test_mask, train_loader, val_loader, test_loader = helper_ppi(args)

    model = get_model(layers=args.layers, in_features=in_features, out_features=out_features, args=args, logger=logger)

    model.searched_arch_op = searched_arch_op
    for key in model.searched_arch_op.keys():
        model.searched_arch_z[key] = [op2z_mapping[key][k] for k in model.searched_arch_op[key]]
    model.searched_arch_z = dict(model.searched_arch_z)
    model.searched_arch_op = dict(model.searched_arch_op)
    model, results = retrain(model, train_loader, val_loader, test_loader, train_mask, val_mask, test_mask, args, logger)
    results['seed'] =seed
    #save_performance_result(args, logger, model)
    return results
def main(dataname,seed,gpu=0):
    parser = argparse.ArgumentParser('Interface for Auto-GNN framework')

    parser.add_argument('--train_ratio', type=float, default=0.6, help='ratio of the train against whole')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio of the val against whole')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='ratio of the test against whole')
    parser.add_argument('--model', type=str, default='Auto-GNN', help='model to use')
    parser.add_argument('--metric', type=str, default='acc', help='metric for evaluating performance', choices=['acc', 'auc'])
    parser.add_argument('--gpu', type=int, default=gpu, help='gpu id')
    parser.add_argument('--directed', type=bool, default=False, help='(Currently unavailable) whether to treat the graph as directed')
    parser.add_argument('--parallel', default=False, action='store_true',
                        help='(Currently unavailable) whether to use multi cpu cores to prepare data')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer to use')

    parser.add_argument('--task', type=str, default='node', help='type of task', choices=['node', 'graph'])
    parser.add_argument('--dataset', type=str, default=dataname, help='dataset name')  # choices=['PROTEINS', 'IMDB-BINARY', 'IMDB-MULTI', 'BZR', 'COX2', 'DD', 'ENZYMES', 'NCI1']
    parser.add_argument('--seed', type=int, default=seed, help='seed to initialize all the random modules')
    parser.add_argument('--clip', type=float, default=0.1, help='gradient clipping')
    # general model and training setting
    parser.add_argument('--epoch', type=int, default=20, help='number of epochs to search')
    parser.add_argument('--retrain_epoch', type=int, default=200, help='number of epochs to retrain')

    parser.add_argument('--layers', type=int, default=2, help='largest number of layers')
    parser.add_argument('--hidden_features', type=int, default=128, help='hidden dimension')
    parser.add_argument('--bs', type=int, default=128, help='minibatch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--l2', type=float, default=1e-4, help='l2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    #parser.add_argument('--dropout', type=float, default=0, help='dropout rate')


    # logging & debug
    parser.add_argument('--log_dir', type=str, default='./log/', help='log directory')

    parser.add_argument('--summary_file', type=str, default='result_summary.log', help='brief summary of training result')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='whether to use debug mode')

    sys_argv = sys.argv
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    check(args)
    logger = set_up_log(args, sys_argv)
    set_random_seed(args)

    if args.dataset != 'PPI':
        dataset, in_features, out_features = get_data(args=args, logger=logger)
        train_mask, val_mask, test_mask = reset_mask(dataset, args=args, logger=logger, stratify=None)
        train_loader, val_loader, test_loader = get_loader(dataset, args=args)
    else:
        in_features, out_features, train_mask, val_mask, test_mask, train_loader, val_loader, test_loader = helper_ppi(args)

    model = get_model(layers=args.layers, in_features=in_features, out_features=out_features, args=args, logger=logger)

    model, results = search(model, train_loader, val_loader, train_mask, val_mask, args, logger)
    model, results = retrain(model, train_loader, val_loader, test_loader, train_mask, val_mask, test_mask, args, logger)

    print(model,dataname,results)
    #save_performance_result(args, logger, model)
if __name__ == '__main__':
    data_list = 'cora citeseer pubmed'.split()#
    for data in data_list:
        main(data,seed=10,gpu=0)
