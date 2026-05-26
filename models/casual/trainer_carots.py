import os

import torch
from tqdm import tqdm

from trainer import Trainer, prepare_inputs
from models.causal.loss import loss_fn
from utils.misc import mkdir
from models.causal.trainer_cuts_plus import DEVICE


class CausalTrainer(Trainer):
    def __init__(self, cfg, model):
        super().__init__(cfg, model)
        self.causal_discoverer_checkpoint_dir = str(mkdir(os.path.join(self.cfg.TRAIN.CHECKPOINT_DIR, self.cfg.CAUSAL_DISCOVERER.lower())))
        self.causal_discoverer_result_dir = self.causal_discoverer_checkpoint_dir
        cfg_causal_discoverer = getattr(self.cfg, f'{self.cfg.CAUSAL_DISCOVERER}')
        cfg_causal_discoverer.TRAIN.CHECKPOINT_DIR = self.causal_discoverer_checkpoint_dir
        cfg_causal_discoverer.RESULT_DIR = self.causal_discoverer_result_dir

    def load_causal_discoverer(self):
        checkpoint_path = os.path.join(self.causal_discoverer_checkpoint_dir, "checkpoint_best.pth")
        
        ckpt = torch.load(checkpoint_path)
        self.model.causal_discoverer.load_state_dict(ckpt['model_state'])
        self.model.causal_discoverer.eval()
        self.model.causal_discoverer.to(DEVICE)
        print("Causal discoverer loaded successfully.")

    def train_causal_discoverer(self):
        from models.causal.trainer_cuts_plus import CUTS_PLUS_Trainer

        trainer_causal_discoverer = CUTS_PLUS_Trainer(self.cfg, self.model.causal_discoverer)
        trainer_causal_discoverer.train()

        self.model.causal_discoverer.load_state_dict(trainer_causal_discoverer.model.state_dict())
        self.model.causal_discoverer.eval()
        self.model.causal_discoverer.to(DEVICE)
        print(f"Causal discoverer ({self.cfg.CAUSAL_DISCOVERER}) trained successfully.")

    def train(self):
        # Attempt to load the causal discoverer, if it fails, train it
        try:
            self.load_causal_discoverer()
        except Exception as e:
            print(f"Failed to load causal discoverer: {e}. Training a new one.")
            self.train_causal_discoverer()

        self.model.positive_augmentor.set_causal_discoverer(self.model.causal_discoverer)

        # Freeze the causal discoverer and positive augmentor
        for param in self.model.causal_discoverer.parameters():
            param.requires_grad = False

        for param in self.model.positive_augmentor.parameters():
            param.requires_grad = False

        metric_best = self.cfg.TRAIN.METRIC_BEST
        for cur_epoch in tqdm(range(self.cfg.SOLVER.START_EPOCH, self.cfg.SOLVER.MAX_EPOCH)):
            # Linearly interpolate SIM_THRESHOLD (避免除以零)
            if self.cfg.Causal.SIM_THRESHOLD_SCHEDULE and self.cfg.SOLVER.MAX_EPOCH > 1:
                self.cfg.Causal.SIM_THRESHOLD = (
                    self.cfg.Causal.SIM_THRESHOLD_START +
                    (self.cfg.Causal.SIM_THRESHOLD_END - self.cfg.Causal.SIM_THRESHOLD_START) *
                    cur_epoch / (self.cfg.SOLVER.MAX_EPOCH - 1)
                )

            # Train the model for one epoch.
            self.train_epoch()

            # Evaluate the model on validation set.
            if self._is_eval_epoch(cur_epoch):
                tracking_meter = self.eval_epoch()
                # check improvement
                is_best = self._check_improvement(tracking_meter.avg, metric_best)
                # Save a checkpoint on improvement.
                if is_best:
                    with open(mkdir(self.cfg.RESULT_DIR) / "best_result.txt", 'w') as f:
                        f.write(f"Val/{tracking_meter.name}: {tracking_meter.avg}\tEpoch: {self.cur_epoch}")
                    print(f"[current best] Val/{tracking_meter.name}: {tracking_meter.avg}\tEpoch: {self.cur_epoch}")
                    self.save_best_model()
                    metric_best = tracking_meter.avg

            self.cur_epoch += 1

        # 训练结束后输出测试集指标 (AUROC, AUPRC, F1)
        self._evaluate_and_print_metrics()

    def _evaluate_and_print_metrics(self):
        """训练结束后在测试集上评估并输出 AUROC, AUPRC, F1"""
        print("\n" + "=" * 60)
        print("在测试集上评估模型...")
        print("=" * 60)

        from models.causal.predictor import Predictor

        # 加载最佳模型
        self.load_best_model()

        # 创建预测器并计算指标
        predictor = Predictor(self.cfg, self.model)
        predictor.predict()

        print("=" * 60)

    def train_step(self, inputs):
        outputs_dict = {}
        inputs, _ = prepare_inputs(inputs)

        if self.cfg.Causal.POSITIVE_AUGMENTOR.ENABLE:
            outputs = self.model(inputs)
        else:
            outputs = self.model(inputs, positive_augment=False)

        loss = loss_fn(outputs, self.cfg)

        self.optimizer.zero_grad()
        loss.backward()
        if self.cfg.SOLVER.GRADIENT_CLIP:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.SOLVER.GRADIENT_CLIP_NORM)
        self.optimizer.step()
        
        outputs_dict["metrics"] = (loss,)
        outputs_dict["losses"] = (loss,)

        return outputs_dict

    @torch.no_grad()
    def eval_step(self, inputs):
        outputs_dict = {}
        inputs, _ = prepare_inputs(inputs)
        
        if self.cfg.Causal.POSITIVE_AUGMENTOR.ENABLE:
            outputs = self.model(inputs)
        else:
            outputs = self.model(inputs, positive_augment=False)
        loss = loss_fn(outputs, self.cfg)

        outputs_dict["metrics"] = (loss,)
        outputs_dict["losses"] = (loss,)

        return outputs_dict

    def load_best_model(self):
        model_path = os.path.join(self.cfg.TRAIN.CHECKPOINT_DIR, "checkpoint_best.pth")
        if os.path.isfile(model_path):
            print(f"Loading checkpoint from {model_path}")
            checkpoint = torch.load(model_path, map_location="cpu")

            state_dict = checkpoint['model_state']
            msg = self.model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == set()
            
            self.model.positive_augmentor.set_causal_discoverer(self.model.causal_discoverer)

            print(f"Loaded pre-trained model from {model_path}")
        else:
            print("=> no checkpoint found at '{}'".format(model_path))

        return self.model