import os, numpy, torch, warnings, glob

def init_system(args):
    warnings.simplefilter("ignore")
    torch.multiprocessing.set_sharing_strategy('file_system')
    args.score_save_path      = os.path.join(args.save_path, 'score.txt')
    args.model_save_path    = os.path.join(args.save_path, 'model')
    os.makedirs(args.model_save_path, exist_ok = True)

    args.modelfiles = glob.glob('%s/model_0*.model'%args.model_save_path)
    args.modelfiles.sort()
    args.score_file = open(args.score_save_path, "a+")
    return args
