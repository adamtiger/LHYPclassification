import torch.optim as opt
import torch
from skimage.transform import resize, rotate
from skimage.util import crop, random_noise
from heartcontour.models.lvhyp.loader3D import LhypDataset, DataType
from torch.utils.data import DataLoader, WeightedRandomSampler
from heartcontour.models.lvhyp.utils import maximum_index_in_saved_weights
from heartcontour.models.lvhyp.utils import HyperParameterSaver
from heartcontour.models.lvhyp.utils import AutoParameterTuner
from heartcontour.models.lvhyp.utils import EarlyStopping
from heartcontour.models.lvhyp import network3d as netw
from heartcontour.models.lvhyp.loss import focal_loss
from heartcontour.utils import print_stdout_logger
from heartcontour.utils import get_logger
from datetime import datetime
import numpy as np
import json
import os


logger = get_logger(__name__)

def run_experiment(config):
    
    # ----  Const parameters ---
    NUM_CROSS_VALS = 3 
    # ----  Main parameters ----
    crop_shape = config['crop_shape']
    resize_shape = config['resize_shape']
    maxangle = config['maxangle']
    device = torch.device('cuda')
    epochs = config['epochs']
    batch_size = config['batch_size']
    criterion = focal_loss
    learning_rate = config['learning_rate']
    opt_gen = config['optimizer']
    optim_lambda = lambda prms: opt_gen(prms, lr=learning_rate)
    sample_folder_path = config['smp_path']
    experiment_output = config['exp_output']

    # ---- Setting the parameters ----
    # All the parameters can be set here
    # therefore they will be committed into
    # the git repo.
    def preprocess(imgs):
        imgs_processed = list()
        for img in imgs:
            vertical_margin = max(0, (img.shape[0] - crop_shape[0]) // 2)
            horizontal_margin = max(0, (img.shape[1] - crop_shape[1]) // 2)
            img = crop(img, ((vertical_margin, vertical_margin), (horizontal_margin, horizontal_margin)))
            img = resize(img, resize_shape)
            imgs_processed.append(img)
        return imgs_processed

    def augmenter(imgs):
        imgs_processed = list()
        angle = 2 * maxangle * np.random.rand() - maxangle
        for img in imgs:
            img = rotate(img, angle)
            random_noise(img, var=0.01)
            imgs_processed.append(img)
        return imgs_processed
    
    def augmenter_for_test(imgs):
        imgs_processed = list()
        for img in imgs:
            imgs_processed.append(img)
        return imgs_processed

    # ---- Function for training and evaluation ----
    def train(model, train_loader, valid_loader, criterion, optim, epochs, root_path):
        logger.info("Started training at {}".format(datetime.now()))
        os.mkdir(root_path)  # creating the folder to save in the results

        sample_num = len(train_loader)
        eval_freq = int(max((sample_num * epochs) // 100, 1))
        save_freq = int(max((sample_num * epochs) // 10, 1))
        global_counter = 0
        cycle_counter = 0
        mean_train_loss = 0

        early_stopping = EarlyStopping(window_size=10, waiting_steps=10)
        should_stop = False

        for epoch in range(epochs):
            for index, sample in enumerate(train_loader):
                global_counter += 1
                predicted = model(sample)
                label = sample['label']
                loss = criterion(predicted, label)

                optim.zero_grad()
                loss.backward()
                optim.step()

                train_loss = loss.cpu().detach().numpy()
                mean_train_loss = (mean_train_loss * cycle_counter + train_loss) / (cycle_counter + 1)

                if global_counter % eval_freq == 0:
                    val_loss = valid(model, valid_loader, criterion, global_counter, root_path)
                    write(
                        os.path.join(root_path, 'train_loss.txt'), 
                        '{} {}'.format(global_counter, mean_train_loss)
                    )
                    mean_train_loss = 0
                    cycle_counter = 0
                
                    #should_stop = early_stopping.add_loss(val_loss)
                    if should_stop:
                        break

                if global_counter % save_freq == 0:
                    model.save(os.path.join(root_path, 'model_state_{}.pt'.format(global_counter)))
                print("\r Current batch {}".format(index), end="")
            if should_stop:
                model.save(os.path.join(root_path, 'model_state_{}.pt'.format(global_counter)))
                break
        print("Training has finished.")
        logger.info("Training has finished {}".format(datetime.now()))

    def valid(model, valid_loader, criterion, counter, root_path):
        mean_loss = 0
        conf_matrix = [0, 0, 0, 0]  # pred - row, label - column -> flatted out row-wise
        for index, sample in enumerate(valid_loader):

            predicted = model(sample)
            label = sample['label']
            loss = criterion(predicted, label)

            loss = loss.cpu().detach().numpy()
            mean_loss = (mean_loss * index + loss) / (index + 1)

            pred = torch.argmax(predicted).cpu().detach().numpy()
            label = label.cpu().detach().numpy()[0]
            conf_matrix[pred * 2 + label] += 1.0

        write(
            os.path.join(root_path, 'val_loss.txt'), 
            '{} {}'.format(counter, mean_loss)
        )

        write(
            os.path.join(root_path, 'conf_mtx.txt'), 
            '{} {} {} {} {} '.format(counter, *conf_matrix)
        )

        return mean_loss  # give back for the early stopping
    
    def test(model, test_loader, root_path):
        for idx in range(20):
            conf_matrix = [0, 0, 0, 0]  # pred - row, label - column -> flatted out row-wise
            for sample in test_loader:
                if np.random.randint(0, 100) > 70:  # uses only a subset of the test samples
                    continue
                predicted = model(sample)
                label = sample['label']

                pred = torch.argmax(predicted).cpu().detach().numpy()
                label = label.cpu().detach().numpy()[0]
                conf_matrix[pred * 2 + label] += 1.0

            write(
                os.path.join(root_path, 'conf_mtx.txt'), 
                '{} {} {} {} {} '.format(idx, *conf_matrix)
            )

            print("\r Current iteration {}".format(idx), end="")

    def write(path, msg):
        with open(path, 'a') as txt:
            txt.write(msg + '\n')

    # ---- Training and test pipeline ----
    # Contains the following phases:
    # 1. 2ch training (with validation results)
    # 2. 4ch training (with validation results)
    # 3. lvot training (with validation results)
    # 4. sa training (with validation results)
    # 5. lvhyp model training (with val. res.)
    # 6. lvhyp model test
    def splitting_data(sample_folder_path, num_cross_val_divides, cross_val_idx, use_cached=False):
        cache_path = os.path.join(experiment_output + str(cross_val_idx), 'cached_splitting.json')
        splitting = None
        if use_cached:
            print_stdout_logger(logger, 'Using cached splitting.')
            with open(cache_path, 'rt') as js:
                splitting = json.load(js)
        else:
            ref_path = '/userhome/student/budai/lhyp_results/20210705/Test21/cached_splitting.json'
            splitting = LhypDataset.split_data2(sample_folder_path, ref_path, num_cross_val_divides)
            with open(cache_path, 'wt') as js:
                json.dump(splitting, js)
        cvi_str = str(cross_val_idx)
        return splitting[cvi_str]['train'], splitting[cvi_str]['valid'], splitting['test']

    def phase_executor(name, model, dtypes, cross_val_idx, use_cached):
        print_stdout_logger(logger, '--- PHASE: {} ---'.format(name))
        print_stdout_logger(logger, 'Splitting dataset ...')
        train_set, valid_set, _ = splitting_data(sample_folder_path, NUM_CROSS_VALS, cross_val_idx, use_cached)
        
        print_stdout_logger(logger, 'Loading samples ...')
        dataset = LhypDataset(
            train_set,
            dtypes,  # list of DataTypes
            preprocess,
            augmenter,
            device
        )
        num_0, num_1 = 0, 0
        for label in dataset.labels:
            if label == 0:
                num_0 += 1
            else:
                num_1 += 1
        weights = [((num_0 + num_1)/num_0 if label==0 else (num_0 + num_1)/num_1) for label in dataset.labels]
        weights = torch.DoubleTensor(weights)
        sampler = WeightedRandomSampler(weights, len(weights))
        tloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, shuffle=False)

        dataset = LhypDataset(
            valid_set,
            dtypes,
            preprocess,
            augmenter_for_test,
            device
        )
        vloader = DataLoader(dataset, 1)
        
        print_stdout_logger(logger, '--- main part ---')
        model = model.to(device)
        train(
            model, 
            tloader, 
            vloader, 
            criterion,
            optim_lambda(model.parameters()),
            epochs,
            os.path.join(experiment_output + str(cross_val_idx), name)
        )

    def run_whole_pretraining(cross_val_idx):
        # Executor pretraining
        print_stdout_logger(logger, 'Executor pretraining ...')
        #phase_executor('phasesa', netw.SAmodel3D(), [DataType.SA], cross_val_idx, False)
        #phase_executor('phase2ch', netw.LA2CHmodel3D(), [DataType.LA2CH], cross_val_idx, True)
        #phase_executor('phase4ch', netw.LA4CHmodel3D(), [DataType.LA4CH], cross_val_idx, True)
        #phase_executor('phaselvot', netw.LALVOTmodel3D(), [DataType.LALVOT], cross_val_idx, True)
        phase_executor('phaselale', netw.LALEmodel3D(), [DataType.LALE], cross_val_idx, True)
        phase_executor('phasesale', netw.SALEmodel3D(), [DataType.SALE], cross_val_idx, True) 

    def ensamble_trainig(paths, cross_val_idx):
        # Ensamble model fine-tuning
        print_stdout_logger(logger, 'Ensamble fine-tuning ...')
        
        model = netw.Lhyp3DModel()
        model.load_pretrained_extractors(paths, device)
        model.freeze_extractors()
        model.to(device)
        phase_executor(
            'phase_lvhyp_model', 
            model, 
            [
                DataType.LA2CH, 
                DataType.LA4CH, 
                DataType.LALVOT,
                DataType.SA,
                DataType.SALE,
                DataType.LALE
            ],
            cross_val_idx,
            True
        )

    def test_lvhyp(path, cross_val_idx):
        # Ensamble model testing
        print_stdout_logger(logger, 'Ensamble model testing ...')
        model = netw.Lhyp3DModel()
        model.load(path, device)

        _, _, test_set = splitting_data(sample_folder_path, NUM_CROSS_VALS, 0, True)
        dataset = LhypDataset(
            test_set,
            [
                DataType.LA2CH, 
                DataType.LA4CH, 
                DataType.LALVOT,
                DataType.SA,
                DataType.SALE,
                DataType.LALE
            ],
            preprocess,
            augmenter_for_test,
            device
        )
        tloader = DataLoader(dataset, 1)

        test_result = os.path.join(experiment_output + str(cross_val_idx), 'final_test')
        os.mkdir(test_result)
        test(model, tloader, test_result)
        print_stdout_logger(logger, 'Ensamble testing was finished')
    
    for cross_val_idx in range(1, NUM_CROSS_VALS):
        experiment_output_cvi = experiment_output + str(cross_val_idx)
        # create output folder
        '''
        if not os.path.exists(experiment_output_cvi):
            os.mkdir(experiment_output_cvi)
        
        # saving
        hps = HyperParameterSaver(os.path.join(experiment_output_cvi, 'hyp_params.json'))
        hps.add_parameter('crop_shape', crop_shape)
        hps.add_parameter('resize_shape', resize_shape)
        hps.add_parameter('maxangle', maxangle)
        hps.add_parameter('epochs', epochs)
        hps.add_parameter('batch_size', batch_size)
        hps.add_parameter('learning_rate', learning_rate)
        hps.add_parameter('optimizer', opt_gen.__name__)
        hps.add_parameter('sample_folder_path', sample_folder_path)
        hps.add_parameter('experiment_output', experiment_output_cvi)
        hps.save()

        # write here the actual execution plan
        '''
        run_whole_pretraining(cross_val_idx)
        
        misw2ch = maximum_index_in_saved_weights(os.path.join(experiment_output_cvi, 'phase2ch'))  # early stopping
        misw4ch = maximum_index_in_saved_weights(os.path.join(experiment_output_cvi, 'phase4ch'))
        miswlvot = maximum_index_in_saved_weights(os.path.join(experiment_output_cvi, 'phaselvot'))
        miswsa = maximum_index_in_saved_weights(os.path.join(experiment_output_cvi, 'phasesa'))
        miswsale = maximum_index_in_saved_weights(os.path.join(experiment_output_cvi, 'phasesale'))
        miswlale = maximum_index_in_saved_weights(os.path.join(experiment_output_cvi, 'phaselale'))
        ensamble_trainig({
            '2ch': os.path.join(experiment_output_cvi, 'phase2ch', misw2ch),
            '4ch': os.path.join(experiment_output_cvi, 'phase4ch', misw4ch),
            'lvot': os.path.join(experiment_output_cvi, 'phaselvot', miswlvot),
            'sa': os.path.join(experiment_output_cvi, 'phasesa', miswsa),
            'sale': os.path.join(experiment_output_cvi, 'phasesale', miswsale),
            'lale': os.path.join(experiment_output_cvi, 'phaselale', miswlale)
        }, cross_val_idx)
        
        misw = maximum_index_in_saved_weights(os.path.join(experiment_output_cvi, 'phase_lvhyp_model'))
        test_lvhyp(os.path.join(experiment_output_cvi, 'phase_lvhyp_model', misw), cross_val_idx)


if __name__ == '__main__':
    from transformer import Sample
    tuner = AutoParameterTuner()
    tuner.add_config('Test1', {
        'crop_shape': [150, 150],
        'resize_shape': [150, 150],
        'maxangle': 8,
        'epochs': 20,
        'batch_size': 16,
        'learning_rate': 5e-4,
        'optimizer': opt.AdamW,
        'smp_path': '/userhome/student/budai/lhyp_data',
        'exp_output': '/userhome/student/budai/lhyp3d_results/20210820/Test1'
    })
    '''
    tuner.add_config('Test2', {
        'crop_shape': [160, 160],
        'resize_shape': [150, 150],
        'maxangle': 8,
        'epochs': 20,
        'batch_size': 16,
        'learning_rate': 5e-4,
        'optimizer': opt.AdamW,
        'smp_path': '/userhome/student/budai/lhyp_data',
        'exp_output': '/userhome/student/budai/lhyp3d_results/20210820/Test2'
    })
    tuner.add_config('Test3', {
        'crop_shape': [150, 150],
        'resize_shape': [150, 150],
        'maxangle': 8,
        'epochs': 20,
        'batch_size': 16,
        'learning_rate': 1e-3,
        'optimizer': opt.Adam,
        'smp_path': '/userhome/student/budai/lhyp_data',
        'exp_output': '/userhome/student/budai/lhyp_results/20210820/Test3'
    })
    tuner.add_config('Test4', {
        'crop_shape': [150, 150],
        'resize_shape': [150, 150],
        'maxangle': 8,
        'epochs': 30,
        'batch_size': 16,
        'learning_rate': 1e-3,
        'optimizer': opt.Adam,
        'smp_path': '/userhome/student/budai/lhyp_data',
        'exp_output': '/userhome/student/budai/lhyp_results/20210820/Test4'
    })
    tuner.add_config('Test5', {
        'crop_shape': [120, 120],
        'resize_shape': [150, 150],
        'maxangle': 8,
        'epochs': 20,
        'batch_size': 16,
        'learning_rate': 1e-3,
        'optimizer': opt.Adam,
        'smp_path': '/userhome/student/budai/lhyp_data',
        'exp_output': '/userhome/student/budai/lhyp_results/20210820/Test5'
    })
    '''
    for config in tuner.next_configuration():
        name, params = config
        print(name)
        run_experiment(params)
