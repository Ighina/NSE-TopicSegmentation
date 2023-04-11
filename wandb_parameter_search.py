import argparse
import json
import os
import re
import sys

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch
import wandb
import yaml

from DataModule import TopSegDataModule
from models.lightning_model import TextSegmenter
#from main_cli import main_cli

class MyParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write('error: %s\n' % message)
            self.print_help()
            sys.exit(2)
            
    
parser = MyParser(
            description = 'Run training with parameters defined in the relative json file')

parser.add_argument("--configuration", "-config", required=True, type=str, help="The file with the hyperparameters to use.")
parser.add_argument("--dataset", "-data", required=True, type=str, help="The dataset to use (must be defined in the DataModule class).")
parser.add_argument("--experiment_name", "-exp", required=True, type=str, help="The name of the current experiment, which is both the ")
parser.add_argument("--blueprint", "-bp", default="configuration.yml", type=str, help="The blueprint to initialize lightning model.")
parser.add_argument('--wandb_key', default="0", type = str, help = "If using wandb, you can directly provide your api key with your option to avoid been prompted to insert the key (good, for example, if you're submitting the script as a job on a cluster)")
parser.add_argument("save_all_checkpoints", action="store_true", help="If included, the programme won't automatically delete the folders related to configurations other than the best one.")

args = parser.parse_args()

# Define sweep config
with open(args.configuration) as f:
    sweep_configuration = json.load(f)

with open(args.blueprint) as f:
    blueprint = yaml.safe_load(f)

# Initialize sweep by passing in config. (Optional) Provide a name of the project.
sweep_id = wandb.sweep(sweep=sweep_configuration, project=args.experiment_name)

data_options = blueprint["data"]

dataloader = TopSegDataModule(**data_options)

def main():
    configuration_ = "_".join([k+"="+v for k, v in wandb.config.items()])
    run = wandb.init(name=configuration_)
    # note that we define values from `wandb.config` instead 
    # of defining hard values
    model_options = blueprint["model"]
    for key, value in wandb.config.items():
        try:
            model_options[key] = value
        except KeyError:
            pass

    trainer_options = blueprint["trainer"]
    for key, value in wandb.config.items():
        try:
            trainer_options[key] = value
        except KeyError:
            pass

    ckpt_dir = os.path.join(args.experiment_name, configuration_)

    # TODO: add the trainer options
    if model_options["metric"]=="Pk" or model_options["metric"]=="WD":
        mode = "min"
    else:
        mode = "max"

    early_stop = EarlyStopping(
              monitor = 'valid_loss',
              patience = blueprint["trainer"]["patience"],
              strict = False,
              verbose = True,
              mode = mode)
            
    checkpoint_callback = ModelCheckpoint(
        monitor='valid_loss',
        dirpath= ckpt_dir,
        filename='checkpoint-{epoch:02d}-{valid_loss:.2f}-{threshold:.2f}',
        save_top_k=1,
        mode=mode,
    )
            
    if model_options["architecture"].architecture.lower().startswith('text'):
        embed_size = dataloader.WordMatrix.weight.shape[1]
    elif model_options["architecture"].architecture.lower().startswith('crossencoder'):
        embed_size = dataloader.encoder
    else:
        embed_size = dataloader.encoder.get_sentence_embedding_dimension()
        
    tagset_size = 2
    
    model = TextSegmenter(architecture = model_options["architecture"],
                            tagset_size = tagset_size, 
                            embedding_dim = embed_size, 
                            hidden_dim = model_options["hidden_unit"], 
                            lr = model_options["learning_rate"], 
                            num_layers = model_options["number_layers"],
                            LSTM = model_options["NoLSTM"],
                            bidirectional = model_options["unidirectional"],
                            loss_fn = model_options["loss_function"],
                            dropout_in = model_options["d_in"],
                            dropout_out = model_options["d_out"],
                            batch_first = model_options["batch_second"],
                            optimizer = blueprint["optimizer"],
                            positional_encoding = model_options["positional_encoding"],
                            nheads = model_options["number_heads"],
                            search_threshold = model_options["search_threshold"],
                            metric = model_options["metric"],
                            end_boundary = model_options["use_end_boundary"],
                            cosine_loss = model_options["cosine_loss"],
                            restricted = model_options["restricted"],
                            window_size = model_options["window_size"],
                            pretrained_embedding_layer = dataloader.WordMatrix,
                            auxiliary_coherence_original = model_options["auxiliary_coherence_original"],
                            restricted_self_attention = model_options["restricted_attention"])
    
    if trainer_options["sixteen_bit"]:
        bits = 16
    else:
        bits = 32
        
    if float(trainer_options["limit_train_batches"])>1.0:
        limit_train_batches = int(trainer_options["limit_train_batches"])
    else:
        limit_train_batches = float(trainer_options["limit_train_batches"])
        
        
    if float(trainer_options["limit_valid_batches"])>1.0:
        limit_valid_batches = int(trainer_options["limit_valid_batches"])
    else:
        limit_valid_batches = float(trainer_options["limit_valid_batches"])
    
    trainer = Trainer(callbacks = [early_stop, checkpoint_callback], 
                        max_epochs = trainer_options["max_epochs"], 
                        gpus = trainer_options["num_gpus"], 
                        auto_lr_find = trainer_options["auto_lr_finder"],
                        gradient_clip_val = trainer_options["gradient_clipping"],
                        precision = bits,
                        detect_anomaly = True,
                        limit_train_batches = limit_train_batches,
                        limit_val_batches = limit_valid_batches)
                        
    if trainer_options["auto_lr_finder"]:
        trainer.tune(model, dataloader)

    trainer.fit(model, dataloader)

    threshold = args.threshold if args.threshold else float(checkpoint_callback.best_model_path.split('=')[-1][:4])

    states = torch.load(ckpt_dir+"best_model")

    validation = states['callbacks'][[x for x in states['callbacks'].keys() if x.startswith("Early")][0]]["best_score"].detach().cpu().tolist()

    model = TextSegmenter.load_from_checkpoint(
                                  checkpoint_callback.best_model_path,
                                  architecture = model_options["architecture"],
                            tagset_size = tagset_size, 
                            embedding_dim = embed_size, 
                            hidden_dim = model_options["hidden_unit"], 
                            lr = model_options["learning_rate"], 
                            num_layers = model_options["number_layers"],
                            LSTM = model_options["NoLSTM"],
                            bidirectional = model_options["unidirectional"],
                            loss_fn = model_options["loss_function"],
                            dropout_in = 0.0,
                            dropout_out = 0.0,
                            batch_first = model_options["batch_second"],
                            optimizer = blueprint["optimizer"],
                            positional_encoding = model_options["positional_encoding"],
                            nheads = model_options["number_heads"],
                            threshold = threshold,
                            metric = model_options["metric"],
                            end_boundary = model_options["use_end_boundary"],
                            cosine_loss = model_options["cosine_loss"],
                            restricted = model_options["restricted"],
                            window_size = model_options["window_size"],
                            pretrained_embedding_layer = dataloader.WordMatrix,
                            auxiliary_coherence_original = model_options["auxiliary_coherence_original"],
                            restricted_self_attention = model_options["restricted_attention"],
                            bootstrap_test = True)
    
    results = trainer.test(model, dataloader)

    if model_options["metric"]=='F1':
        f1_label = 'test_loss'
        pk_label = 'Pk_loss'
        wd_label = 'WD_loss'
    elif model_options["metric"]=='WD':
        f1_label = 'F1_loss'
        pk_label = 'Pk_loss'
        wd_label = 'test_loss'
    elif model_options["metric"]=='B':
        f1_label = 'b_f1'
        pk_label = 'b_precision'
        wd_label = 'b_recall'
        test_label = 'test_loss'
    
    elif model_options["metric"]=='scaiano':
        f1_label = 'test_loss'
        pk_label = 'b_precision'
        wd_label = 'b_recall'
        
    else:
        f1_label = 'F1_loss'
        pk_label = 'test_loss'
        wd_label = 'WD_loss'
    
    if model_options["metric"] == 'B' or model_options["metric"] == 'scaiano':
        with open(os.path.join(ckpt_dir, 'results.txt'), 'w') as f:
            f.write('Results for experiment {} with following parameters:'.format(args.experiment_name))
            f.write('Sentence encoder: {}\n'.format(data_options["encoder"]))
            f.write('Neural architecture: {}\n'.format(model_options["architecture"]))
            f.write('Batch size: {}\n'.format(trainer_options["batch_size"]))
            f.write('Hidden units: {}\n'.format(model_options["hidden_unit"]))
            f.write('Number of layers: {}\n'.format(model_options["number_layers"]))
            f.write('Optimizer: {}'.format(trainer_options["optimizer"]))
            f.write('Threshold: {}'.format(threshold))
            f.write('B_precision score: {}\n'.format(results[0][pk_label]))
            f.write('B_recall score: {}\n'.format(results[0][wd_label]))
            f.write('B_F1 score: {}\n'.format(results[0][f1_label]))
            f.write('Validation loss: {}\n'.format(validation))
            if model_options["metric"]=='B': 
                f.write('B Similarity score: {}\n'.format(results[0][test_label]))
    else:
        with open(os.path.join(ckpt_dir, 'results.txt'), 'w') as f:
            f.write('Results for experiment {} with following parameters:'.format(args.experiment_name))
            f.write('Sentence encoder: {}\n'.format(data_options["encoder"]))
            f.write('Neural architecture: {}\n'.format(model_options["architecture"]))
            f.write('Batch size: {}\n'.format(trainer_options["batch_size"]))
            f.write('Hidden units: {}\n'.format(model_options["hidden_unit"]))
            f.write('Number of layers: {}\n'.format(model_options["number_layers"]))
            f.write('Optimizer: {}'.format(trainer_options["optimizer"]))
            f.write('Threshold: {}'.format(threshold))
            f.write('PK score: {}\n'.format(results[0][pk_label]))
            f.write('WD score: {}\n'.format(results[0][wd_label]))
            f.write('F1 score: {}\n'.format(results[0][f1_label]))
            f.write('Validation loss: {}\n'.format(validation))

wandb.agent.sweep(sweep_id, main) # TODO: CHECK THIS, JUST A PLACEHOLDER

if blueprint["model"]["metric"]=="Pk" or blueprint["model"]["metric"]=="WD":
    mult = -1
else:
    mult = 1

best_valid = -1
best_dirc = None
for dirc in os.listdir(args.experiment):
    with open(os.path.join(args.experiment, dirc, "results.txt")) as f:
        lines = f.readlines()
        valid = float(re.findall("[0]?[\.]?\d+", lines[-1])[0])*mult
        if valid>best_valid:
            best_valid=valid
            best_dirc = dirc

if best_dirc is not None:
    os.rename(os.path.join(args.experiment, best_dirc), os.path.join(args.experiment, "best_configuration"))
