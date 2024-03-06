from lag_llama.gluon.estimator import LagLlamaEstimator
from lag_transformer.gluon.estimator import LagTransformerEstimator

from informer.estimator import InformerEstimator
from etsformer.estimator import ETSformerEstimator
from ns_transformer.estimator import NSTransformerEstimator
from hopfield.estimator import HopfieldEstimator
from switch.estimator import SwitchTransformerEstimator
from reformer.estimator import ReformerEstimator
from perceiverar.estimator import PerceiverAREstimator
# from xformer.estimator import XformerEstimator


def get_model_estimator(args, train_args):
    # print("train_args", train_args)
    aug_args = {
            "aug_prob":args.aug_prob,
            "freq_mask_rate":args.freq_mask_rate,
            "freq_mixing_rate":args.freq_mixing_rate,
            "jitter_prob":args.jitter_prob,
            "jitter_sigma":args.jitter_sigma,
            "scaling_prob":args.scaling_prob,
            "scaling_sigma":args.scaling_sigma,
            "rotation_prob":args.rotation_prob,
            "permutation_prob":args.permutation_prob,
            "permutation_max_segments":args.permutation_max_segments,
            "permutation_seg_mode":args.permutation_seg_mode,
            "magnitude_warp_prob":args.magnitude_warp_prob,
            "magnitude_warp_sigma":args.magnitude_warp_sigma,
            "magnitude_warp_knot":args.magnitude_warp_knot,
            "time_warp_prob":args.time_warp_prob,
            "time_warp_sigma":args.time_warp_sigma,
            "time_warp_knot":args.time_warp_knot,
            "window_slice_prob":args.window_slice_prob,
            "window_slice_reduce_ratio":args.window_slice_reduce_ratio,
            "window_warp_prob":args.window_warp_prob,
            "window_warp_window_ratio":args.window_warp_window_ratio,
            "window_warp_scales":args.window_warp_scales,
    }

    other_args = {
            "prediction_length":args.prediction_length,
            "context_length":args.context_length,
            "input_size":1,
            "batch_size":args.batch_size,
            "scaling":args.data_normalization, # TODO: fix naming collision with scaling augmentations
            "lr":args.lr,
            "weight_decay":args.weight_decay,
            "distr_output":args.distr_output,
            "num_batches_per_epoch":args.num_batches_per_epoch,
            "num_parallel_samples":args.num_parallel_samples,
            "time_feat":args.time_feat,
            "dropout":args.dropout,
    }

    train_args.update(other_args)
    if args.model == "informer":
        model_args = {
        "num_encoder_layers": args.num_encoder_layer, #
        "num_decoder_layers": args.n_layer,
        "dim_feedforward" : args.dim_feedforward,
        "d_model": args.n_embd_per_head * args.n_head,
        "nhead": args.n_head,
        }
        model_args.update(train_args)
        estimator = InformerEstimator(**model_args)
    elif args.model == "lag_llama":
        train_args.update(aug_args)
        
        model_args = {"n_layer":args.n_layer,
            "n_embd_per_head":args.n_embd_per_head,
            "n_head":args.n_head,
            "max_context_length":2048,
            "rope_scaling":None}
        model_args.update(train_args)
        estimator = LagLlamaEstimator(**model_args)
    elif args.model == "etsformer":
        model_args = {
        "model_dim": args.dim_feedforward, 
        "k_largest_amplitudes": args.k_largest_amplitudes, #
        "embed_kernel_size": args.embed_kernel_size, #
        "nhead": args.n_head,
        "num_layers": args.n_layer,
        "d_model": args.n_embd_per_head * args.n_head,
        }
        model_args.update(train_args)
        estimator = ETSformerEstimator(**model_args)
    elif args.model == "hopfield":
        model_args = {
        "num_encoder_layers": args.num_encoder_layer,
        "num_decoder_layers": args.n_layer,
        "dim_feedforward": args.dim_feedforward,
        "d_model": args.n_embd_per_head * args.n_head,
        "nhead": args.n_head,
        }
        model_args.update(train_args)
        estimator = HopfieldEstimator(**model_args)
    elif args.model == "nstransformer":
        model_args = {
        "nhead": args.n_head,
        "num_encoder_layers": args.num_encoder_layer,
        "num_decoder_layers": args.n_layer,
        "dim_feedforward": args.dim_feedforward,
        "d_model": args.n_embd_per_head * args.n_head,
        }
        model_args.update(train_args)
        estimator = NSTransformerEstimator(**model_args)
    elif args.model == "switch":
        model_args = {
        "nhead": args.n_head,
        "num_encoder_layers": args.num_encoder_layer,#
        "num_decoder_layers": args.n_layer,
        "dim_feedforward": args.dim_feedforward,
        "capacity_factor": args.capacity_factor,#
        "n_experts": args.num_expert,#
        "d_model": args.n_embd_per_head * args.n_head,
        "is_scale_prob": True,
        "drop_tokens":  False,
        }
        model_args.update(train_args)
        estimator = SwitchTransformerEstimator(**model_args)
    elif args.model == "reformer":
        model_args = {
            "num_encoder_layers": args.num_encoder_layer,#
            "num_decoder_layers": args.n_layer,
            "d_model": args.n_embd_per_head * args.n_head,
            "nhead": args.n_head,
        }
        model_args.update(train_args)
        estimator = ReformerEstimator(**model_args)
    elif args.model == "perceiver_ar":
        model_args = {
            "depth": args.depth,#
            "perceive_depth": args.perceive_depth,#
            "heads": args.n_head,
            "hidden_size": args.hidden_size,#
            "dropout": args.dropout,
            "cross_attn_dropout": args.cross_attn_dropout,#
            "perceive_max_heads_process": args.perceive_max_heads_process,#
            "ff_mult": args.ff_mult,
        }
        model_args.update(train_args)
        estimator = PerceiverAREstimator(**model_args)
    else:
        train_args.update(aug_args)
        model_args = {"num_encoder_layers":args.num_encoder_layer,
            "num_decoder_layers":args.n_layer,
            "dim_feedforward":args.dim_feedforward,
            "d_model_per_head":args.n_embd_per_head,
            "nhead":args.n_head,}
        model_args.update(train_args)
        estimator = LagTransformerEstimator(**model_args)
    # else:
    #     model_args = {
    #         "num_encoder_layers": args.num_encoder_layer,#
    #         "num_decoder_layers": args.n_layer,
    #         "d_model": args.n_embd_per_head * args.n_head,
    #         "nhead": args.n_head,
    #         "hidden_layer_multiplier": args.hidden_layer_multiplier,
    #     }
    #     model_args.update(train_args)
    #     estimator = XformerEstimator(**model_args)
    return estimator
    
def get_hp_params(trial, model) -> dict:
    train_args = {
        "context_length": trial.suggest_int("context_length", 128, 320, 64),
        "dropout": trial.suggest_float('dropout', 0.0, 0.5),
        "weight_decay": trial.suggest_float('weight_decay', 0.0, 0.001),
        "lr":trial.suggest_loguniform('learning_rate', 1e-5, 1e-3),
        "stratified_sampling": trial.suggest_categorical("stratified_sampling", ["series", "timesteps"]),
        "data_normalization": trial.suggest_categorical("data_normalization", ["mean", "std", "robust"]),}
    if model=="informer":
        train_args.update({
        "dim_feedforward": trial.suggest_categorical("dim_feedforward", [256, 512, 1024, 2048]),   
        # "batch_size": trial.suggest_int("batch_size", 128, 256, 64),
        "num_encoder_layers": trial.suggest_categorical("num_encoder_layers", [2,3,4,6]),
        "num_decoder_layers": trial.suggest_int("num_decoder_layers", 2, 16,4),
        "d_model": trial.suggest_categorical('d_model', [64, 256, 512]),
        "nhead": trial.suggest_categorical('nhead', [2, 4, 8]),
        "time_feat": trial.suggest_categorical('time_feat', [True, False]),
        })
        return train_args
    elif model=="etsformer":
        train_args.update({
        "model_dim": trial.suggest_categorical("model_dim", [256, 512, 1024, 2048]),   
        "num_layers": trial.suggest_int("num_layers", 2, 16,4),
        "k_largest_amplitudes": trial.suggest_int("k_largest_amplitudes", 2, 16,4),
        "embed_kernel_size": trial.suggest_int("embed_kernel_size", 2, 16,4),
        "d_model": trial.suggest_categorical('d_model', [64, 256, 512]),
        "nhead": trial.suggest_categorical('nhead', [2, 4, 8]),
        "time_feat": trial.suggest_categorical('time_feat', [True, False]),
        })
        return train_args
    elif model=="hopfield":
        train_args.update({
        "num_encoder_layers": trial.suggest_int("num_encoder_layers", 2, 16,4),
        "num_decoder_layers": trial.suggest_int("num_decoder_layers", 2, 16,4),
        "dim_feedforward": trial.suggest_categorical("dim_feedforward", [256, 512, 1024, 2048]),
        "d_model": trial.suggest_categorical('d_model', [64, 256, 512]),
        "nhead": trial.suggest_categorical('nhead', [2, 4, 8]),
        "lr":trial.suggest_loguniform('learning_rate', 1e-5, 1e-3),
        })
        return train_args
    elif model=="nstransformer":
        train_args.update({
        "nhead": trial.suggest_categorical('nhead', [2, 4, 8]),
        "num_encoder_layers": trial.suggest_int("num_encoder_layers", 2, 16,4),
        "num_decoder_layers": trial.suggest_int("num_decoder_layers", 2, 16,4),
        "dim_feedforward": trial.suggest_categorical("dim_feedforward", [256, 512, 1024, 2048]),
        "d_model": trial.suggest_categorical('d_model', [64, 256, 512]),
         "time_feat": trial.suggest_categorical('time_feat', [True, False]),
        })
        return train_args
    elif model=="switch":
        train_args.update({
        "nhead": trial.suggest_categorical('nhead', [2, 4, 8]),
        "num_encoder_layers": trial.suggest_int("num_encoder_layers", 2, 16,4),
        "num_decoder_layers": trial.suggest_int("num_decoder_layers", 2, 16,4),
        "dim_feedforward": trial.suggest_categorical("dim_feedforward", [256, 512, 1024, 2048]),
        "d_model": trial.suggest_categorical('d_model', [64, 256, 512]),
        "n_experts": trial.suggest_categorical('n_experts', [4, 16, 32, 64]),
        "capacity_factor": trial.suggest_categorical('capacity_factor', [0,1,2]),
         "time_feat": trial.suggest_categorical('time_feat', [True, False]),
        })
        return train_args
    elif model=="reformer":
        train_args.update({
        "context_length": trial.suggest_int("context_length", 128, 320, 64),
        "nhead": trial.suggest_categorical('nhead', [2, 4, 8]),
        "num_encoder_layers": trial.suggest_int("num_encoder_layers", 2, 16,4),
        "num_decoder_layers": trial.suggest_int("num_decoder_layers", 2, 16,4),
        "d_model": trial.suggest_categorical('d_model', [64, 256, 512]),
         "lr":trial.suggest_loguniform('learning_rate', 1e-5, 1e-3),
         "time_feat": trial.suggest_categorical('time_feat', [True, False]),
        })
        return train_args
    elif model=="perceiver_ar":
        train_args.update({
        "num_encoder_layers": trial.suggest_int("num_encoder_layers", 2, 16,4),
        "num_decoder_layers": trial.suggest_int("num_decoder_layers", 2, 16,4),
        "d_model": trial.suggest_categorical('d_model', [64, 256, 512]),
        "nhead": trial.suggest_categorical('nhead', [2, 4, 8]),
         "time_feat": trial.suggest_categorical('time_feat', [True, False]),
        })
        return train_args
    else:#lag_transformer
        train_args.update({
        "num_decoder_layers": trial.suggest_int("num_decoder_layers", 2, 16,4),
        "dim_feedforward": trial.suggest_categorical("dim_feedforward", [256, 512, 1024, 2048]),
        "d_model_per_head":trial.suggest_categorical('d_model_per_head', [64, 256, 512]),
        "nhead": trial.suggest_categorical('nhead', [2, 4, 8]),
        "time_feat": trial.suggest_categorical('time_feat', [True, False]),
        })
        return train_args

def get_hp_params_estimator(args, train_args, params):
    aug_args = {
            "aug_prob":args.aug_prob,
            "freq_mask_rate":args.freq_mask_rate,
            "freq_mixing_rate":args.freq_mixing_rate,
            "jitter_prob":args.jitter_prob,
            "jitter_sigma":args.jitter_sigma,
            "scaling_prob":args.scaling_prob,
            "scaling_sigma":args.scaling_sigma,
            "rotation_prob":args.rotation_prob,
            "permutation_prob":args.permutation_prob,
            "permutation_max_segments":args.permutation_max_segments,
            "permutation_seg_mode":args.permutation_seg_mode,
            "magnitude_warp_prob":args.magnitude_warp_prob,
            "magnitude_warp_sigma":args.magnitude_warp_sigma,
            "magnitude_warp_knot":args.magnitude_warp_knot,
            "time_warp_prob":args.time_warp_prob,
            "time_warp_sigma":args.time_warp_sigma,
            "time_warp_knot":args.time_warp_knot,
            "window_slice_prob":args.window_slice_prob,
            "window_slice_reduce_ratio":args.window_slice_reduce_ratio,
            "window_warp_prob":args.window_warp_prob,
            "window_warp_window_ratio":args.window_warp_window_ratio,
            "window_warp_scales":args.window_warp_scales,
    }

    other_args = {
            "prediction_length":args.prediction_length,
            "context_length":params["context_length"],
            "input_size":1,
            "batch_size":args.batch_size,
            "scaling":params["data_normalization"], # TODO: fix naming collision with scaling augmentations
            "lr":params["lr"],
            "weight_decay":params["weight_decay"],
            "distr_output":args.distr_output,
            "num_batches_per_epoch":args.num_batches_per_epoch,
            "num_parallel_samples":args.num_parallel_samples,
            "time_feat":params["time_feat"],
            "dropout":params["dropout"],
    }

    train_args.update(other_args)
    if args.model == "informer":
        model_args = {
        "num_encoder_layers": params["num_encoder_layers"], #
        "num_decoder_layers": params["num_decoder_layers"],
        "dim_feedforward" : params["dim_feedforward"],
        "d_model": params["d_model"],
        "nhead": params["nhead"],
        }
        model_args.update(train_args)
        estimator = InformerEstimator(**model_args)

    elif args.model == "etsformer":
        model_args = {
        "model_dim": params["dim_feedforward"], 
        "k_largest_amplitudes": params["k_largest_amplitudes"], #
        "embed_kernel_size": params["embed_kernel_size"], #
        "nhead": params["nhead"],
        "num_layers": params["num_layers"],
        "d_model": params["d_model"],
        }
        model_args.update(train_args)
        estimator = ETSformerEstimator(**model_args)
    elif args.model == "hopfield":
        model_args = {
        "num_encoder_layers": params["num_encoder_layers"],
        "num_decoder_layers": params["num_decoder_layers"],
        "dim_feedforward": params["dim_feedforward"],
        "d_model": params["d_model"],
        "nhead": params["nhead"],
        }
        model_args.update(train_args)
        estimator = HopfieldEstimator(**model_args)
    elif args.model == "nstransformer":
        model_args = {
        "nhead": params["nhead"],
        "num_encoder_layers": params["num_encoder_layer"],
        "num_decoder_layers": params["num_decoder_layers"],
        "dim_feedforward": params["dim_feedforward"],
        "d_model": params["d_model"],
        }
        model_args.update(train_args)
        estimator = NSTransformerEstimator(**model_args)
    elif args.model == "switch":
        model_args = {
        "nhead": params["nhead"],
        "num_encoder_layers": params["num_encoder_layers"],
        "num_decoder_layers": params["num_decoder_layers"],
        "dim_feedforward": params["dim_feedforward"],
        "capacity_factor": params["capacity_factor"],#
        "n_experts": params["num_expert"],#
        "d_model": params["d_model"],
        "is_scale_prob": True,
        "drop_tokens":  False,
        }
        model_args.update(train_args)
        estimator = SwitchTransformerEstimator(**model_args)
    elif args.model == "reformer":
        model_args = {
            "num_encoder_layers": params["num_encoder_layers"],
            "num_decoder_layers": params["num_decoder_layers"],
            "d_model": params["d_model"],
            "nhead": params["nhead"],
        }
        model_args.update(train_args)
        estimator = ReformerEstimator(**model_args)
    elif args.model == "perceiver_ar":
        model_args = {
            "depth": params["depth"],#
            "perceive_depth": params["perceive_depth"],#
            "heads": params["n_head"],
            "hidden_size": params["hidden_size"],#
            "dropout": params["dropout"],
            "cross_attn_dropout": params["cross_attn_dropout"],#
            "perceive_max_heads_process": params["perceive_max_heads_process"],#
            "ff_mult": params["ff_mult"],
        }
        model_args.update(train_args)
        estimator = PerceiverAREstimator(**model_args)
    else:
        train_args.update(aug_args)
        model_args = {"num_encoder_layers":params["num_encoder_layers"],
            "num_decoder_layers":params["num_decoder_layers"],
            "dim_feedforward":params["dim_feedforward"],
            "d_model_per_head":params["n_embd_per_head"],
            "nhead":params["n_head"],}
        model_args.update(train_args)
        estimator = LagTransformerEstimator(**model_args)
    return estimator

