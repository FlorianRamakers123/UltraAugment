from pydoc import locate
import torch

from monai.engines import SupervisedEvaluator, SupervisedTrainer, IterationEvents
from monai.handlers import from_engine, ValidationHandler, StatsHandler, TensorBoardStatsHandler, \
    CheckpointSaver
from monai.transforms import Compose
from monai.utils import CommonKeys
from torch.utils.tensorboard import SummaryWriter

from config import ConfigKeys
from dataloader import get_dataloaders, _construct_transform_chain


def _trainer_iteration_update(engine: SupervisedTrainer, batchdata: dict[str, torch.Tensor]) -> dict:
    if batchdata is None:
        raise ValueError("Must provide batch data for current iteration.")
    batch = engine.prepare_batch(batchdata, engine.state.device, engine.non_blocking, **engine.to_kwargs)
    model_input = torch.cat([batch[key] for key in engine.config[ConfigKeys.MODEL_INPUT_KEYS]], dim=1).to(
        engine.state.device)
    engine.state.output = batch

    def _compute_pred_loss(is_closure: bool = False):
        if engine.config[ConfigKeys.USE_CHECKPOINTING]:
            outputs = torch.utils.checkpoint.checkpoint(engine.network, model_input)
        else:
            outputs = engine.inferer(model_input, engine.network)

        if not is_closure:
            engine.state.output[CommonKeys.PRED] = outputs
            engine.fire_event(IterationEvents.FORWARD_COMPLETED)

        loss = 0
        for loss_func, info in zip(engine.loss_function, engine.config[ConfigKeys.LOSS_FUNC]):
            w = info.get("weight", 1.0)
            name = info.get("name", info["class"].split(".")[-1])

            if engine.config[ConfigKeys.DEEP_SUPERVISION] and info.get("deepsupervision", False):
                ds_outputs = torch.unbind(outputs, dim=1)
                loss_comp = 0
                for i, output_ir in enumerate(ds_outputs):
                    engine.state.output[CommonKeys.PRED] = output_ir
                    loss_inputs = [engine.state.output[key] for key in info["input_keys"]]
                    loss_comp += 0.5 ** i * loss_func(*loss_inputs).mean()
                engine.state.output[CommonKeys.PRED] = ds_outputs[0]
            else:
                if engine.config[ConfigKeys.DEEP_SUPERVISION]:
                    engine.state.output[CommonKeys.PRED] = torch.unbind(outputs, dim=1)[0]
                loss_inputs = [engine.state.output[key] for key in info["input_keys"]]
                loss_comp = loss_func(*loss_inputs).mean()
            if not is_closure:
                engine.summary_writer.add_scalar(name, loss_comp.item(), engine.state.epoch)
            loss += w * loss_comp
        if not is_closure:
            engine.fire_event(IterationEvents.LOSS_COMPLETED)
        return loss

    engine.network.train()
    engine.optimizer.zero_grad(set_to_none=engine.optim_set_to_none)

    if engine.amp and engine.scaler is not None:
        with torch.cuda.amp.autocast(**engine.amp_kwargs):
            _compute_pred_loss()
        engine.scaler.scale(engine.state.output[CommonKeys.LOSS]).backward()
        engine.fire_event(IterationEvents.BACKWARD_COMPLETED)
        engine.scaler.step(engine.optimizer)
        engine.scaler.update()
    else:
        loss = _compute_pred_loss()
        engine.state.output[CommonKeys.LOSS] = loss
        loss.backward()
        engine.fire_event(IterationEvents.BACKWARD_COMPLETED)
        engine.optimizer.step()  # lambda: _compute_pred_loss(True))
    engine.fire_event(IterationEvents.MODEL_COMPLETED)

    return engine.state.output


def _evaluator_iteration_update(engine: SupervisedEvaluator, batchdata: dict[str, torch.Tensor]) -> dict:
    if batchdata is None:
        raise ValueError("Must provide batch data for current iteration.")
    batch = engine.prepare_batch(batchdata, engine.state.device, engine.non_blocking, **engine.to_kwargs)
    model_input = torch.cat([batch[key] for key in engine.config[ConfigKeys.MODEL_INPUT_KEYS]], dim=1).to(
        engine.state.device)
    engine.state.output = batch

    # execute forward computation
    with engine.mode(engine.network):
        if engine.amp:
            with torch.cuda.amp.autocast(**engine.amp_kwargs):
                if isinstance(engine.config[ConfigKeys.PRED_OUTPUT_KEYS], str):
                    engine.state.output[engine.config[ConfigKeys.PRED_OUTPUT_KEYS]] = engine.inferer(model_input,
                                                                                                     engine.network)
                else:
                    output = engine.inferer(model_input, engine.network)
                    for pred_key, pred in zip(engine.config[ConfigKeys.PRED_OUTPUT_KEYS], output):
                        engine.state.output[pred_key] = pred
        else:
            if isinstance(engine.config[ConfigKeys.PRED_OUTPUT_KEYS], str):
                engine.state.output[engine.config[ConfigKeys.PRED_OUTPUT_KEYS]] = engine.inferer(model_input,
                                                                                                 engine.network)
            else:
                output = engine.inferer(model_input, engine.network)
                for pred_key, pred in zip(engine.config[ConfigKeys.PRED_OUTPUT_KEYS], output):
                    engine.state.output[pred_key] = pred
    engine.fire_event(IterationEvents.FORWARD_COMPLETED)
    engine.fire_event(IterationEvents.MODEL_COMPLETED)

    return engine.state.output


def get_network(config):
    device = torch.device(config[ConfigKeys.DEVICE])
    network_class = locate(config[ConfigKeys.NETWORK_CLASS])
    model = network_class(**config[ConfigKeys.NETWORK_ARGS]).to(device)
    return model

def create_metrics(metrics_description):
    if metrics_description is None:
        return None
    metrics = {}
    for metric_name, desc in metrics_description.items():
        metric_class = locate(desc["class"])
        metrics[metric_name] = metric_class(**desc["args"], output_transform=from_engine(desc["input_keys"]))
    return metrics

def create_optimizer(model, config):
    optim_class = locate(config[ConfigKeys.OPTIMIZER_CLASS])
    optimizer = optim_class(model.parameters(), **config[ConfigKeys.OPTIMIZER_ARGS])
    return optimizer

def create_loss_function(config):
    loss_func = []
    for loss_info in config[ConfigKeys.LOSS_FUNC]:
        loss_function_class = locate(loss_info["class"])
        kwargs = loss_info.get("args", {})
        loss_f = loss_function_class(**kwargs)
        loss_func.append(loss_f)
    return loss_func


def setup_train_val_test_env(run_dir, config, l, dataloaders=None):
    device = torch.device(config[ConfigKeys.DEVICE])
    model = get_network(config)
    if dataloaders is None:
        train_loader, val_loader, test_loader = get_dataloaders(config)
    else:
        train_loader, val_loader, test_loader = dataloaders

    optimizer = create_optimizer(model, config)
    loss_func = create_loss_function(config)

    to_device = lambda x, device: x.to(device) if isinstance(x, torch.Tensor) else x

    def prepare_batch(batch, device, *_):
        return {key: to_device(batch[key], device) for key in batch}

    summary_writer = SummaryWriter(log_dir=run_dir)

    validation_handlers = [
        StatsHandler(output_transform=lambda x: None),
        TensorBoardStatsHandler(summary_writer=summary_writer, log_dir=run_dir, output_transform=lambda x: None),
    ]
    if config[ConfigKeys.VALIDATION_KEY_METRIC]:
        validation_handlers.append(CheckpointSaver(save_dir=run_dir, save_dict={"net": model},
                                                   key_metric_name=config[ConfigKeys.VALIDATION_KEY_METRIC],
                                                   save_key_metric=True,
                                                   file_prefix="model",
                                                   key_metric_negative_sign=config[ConfigKeys.SMALLEST_VAL_METRIC]))

    train_metrics = create_metrics(config[ConfigKeys.TRAIN_METRICS])
    validation_metrics = create_metrics(config[ConfigKeys.VALIDATION_METRICS])
    test_metrics = create_metrics(config[ConfigKeys.TEST_METRICS])

    posttransforms = _construct_transform_chain(config, config[ConfigKeys.POSTTRANSFORMS])
    posttransforms_train = Compose(posttransforms["train"]).set_random_state(seed=config[ConfigKeys.SEED])
    posttransforms_val = Compose(posttransforms["validation"]).set_random_state(seed=config[ConfigKeys.SEED])
    posttransforms_test = Compose(posttransforms["test"]).set_random_state(seed=config[ConfigKeys.SEED])

    smaller = lambda c, p: c < p
    greater = lambda c, p: c > p

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=model,
        key_val_metric=validation_metrics,
        metric_cmp_fn=smaller if config[ConfigKeys.SMALLEST_VAL_METRIC] else greater,
        val_handlers=validation_handlers,
        postprocessing=posttransforms_val,
        prepare_batch=prepare_batch,
        iteration_update=_evaluator_iteration_update
    )
    evaluator.logger = l
    evaluator.config = config
    train_handlers = [
        ValidationHandler(validator=evaluator, interval=config[ConfigKeys.VALIDATION_INTERVAL], epoch_level=True),
        StatsHandler(tag_name="train_loss", output_transform=from_engine([CommonKeys.LOSS], first=True)),
        TensorBoardStatsHandler(log_dir=run_dir, tag_name="train_loss",
                                output_transform=from_engine([CommonKeys.LOSS], first=True)),
        CheckpointSaver(save_dir=run_dir, save_dict={"model": model, "opt": optimizer},
                        save_interval=config[ConfigKeys.TRAINING_CHECKPOINT_INTERVAL], file_prefix="train"),
    ]

    trainer = SupervisedTrainer(
        device=device,
        max_epochs=config[ConfigKeys.NUM_EPOCHS],
        train_data_loader=train_loader,
        network=model,
        optimizer=optimizer,
        loss_function=loss_func,
        train_handlers=train_handlers,
        key_train_metric=train_metrics,
        metric_cmp_fn=smaller if config[ConfigKeys.SMALLEST_VAL_METRIC] else greater,
        postprocessing=posttransforms_train,
        prepare_batch=prepare_batch,
        iteration_update=_trainer_iteration_update
    )
    trainer.logger = l
    trainer.config = config
    trainer.summary_writer = summary_writer
    tester = SupervisedEvaluator(
        device=device,
        val_data_loader=test_loader,
        network=model,
        postprocessing=posttransforms_test,
        prepare_batch=prepare_batch,
        iteration_update=_evaluator_iteration_update,
        key_val_metric=test_metrics
    )
    tester.logger = l
    tester.config = config
    return trainer, evaluator, tester, summary_writer
