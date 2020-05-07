def set_template(args):
    if args.template == 'DBVSR':
        args.model = "DBVSR"
        args.save = "dbvsr_test"
        args.data_train = 'VideoSR'
        args.dir_data = '/home/jinshan/data3/CSS_code_Kernel/code_test/datasets/REDS/test'
        args.data_test = 'VideoSR'
        args.testset = "REDS4"
        args.dir_data_test = '/home/jinshan/data3/CSS_code_Kernel/code_test/datasets/REDS/test'
        args.fc_pretrain = '../pretrain/kernel.pt'
        args.pwc_pretrain = '../pretrain/network-default.pytorch'
        args.fcn_number = 2
        args.kernel_size = (15, 15)
        args.grad_clip = 0.5
        # args.save_middle_models = True
        args.test_model_path = "../models_in_paper/dbvsr/model_dbvsr.pt"
        # args.test_only = True

    elif args.template == 'baseline_lr':
        args.model = "baseline_lr"
        args.save = "baseline_lr_test"
        args.data_train = 'VideoSR'
        args.dir_data = '/home/jinshan/data3/CSS_code_Kernel/code_test/datasets/REDS/test'
        args.data_test = 'VideoSR'
        args.testset = "REDS4"
        args.dir_data_test = '/home/jinshan/data3/CSS_code_Kernel/code_test/datasets/REDS/test'
        args.fc_pretrain = '../pretrain/kernel.pt'
        args.pwc_pretrain = '../pretrain/network-default.pytorch'
        args.fcn_number = 2
        args.kernel_size = (15, 15)
        args.grad_clip = 0.5
        args.test_model_path = "../models_in_paper/baseline_lr/model_lr.pt"
        # args.test_only = True

    elif args.template == 'baseline_hr':
        args.model = "baseline_hr"
        args.save = "baseline_hr_test"
        args.data_train = 'VideoSR'
        # args.dir_data = '../datasets/REDS/train'
        args.dir_data = '/home/jinshan/data3/CSS_code_Kernel/code_test/datasets/REDS/test'
        args.data_test = 'VideoSR'
        args.testset = "REDS4"
        args.dir_data_test = '/home/jinshan/data3/CSS_code_Kernel/code_test/datasets/REDS/test'
        args.fc_pretrain = '../pretrain/kernel.pt'
        args.pwc_pretrain = '../pretrain/network-default.pytorch'
        args.fcn_number = 2
        args.kernel_size = (15, 15)
        args.grad_clip = 0.5
        args.test_model_path = "../models_in_paper/baseline_hr/model_hr.pt"
        # args.test_only = True
    else:
        raise NotImplementedError('Template [{:s}] is not found'.format(args.template))
