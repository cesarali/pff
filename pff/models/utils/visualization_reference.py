from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch

try:  # Support both modern and legacy Lightning imports
    import lightning.pytorch as pl
except Exception:  # pragma: no cover
    import pytorch_lightning as pl  # type: ignore

if TYPE_CHECKING:  # pragma: no cover
    from torchtyping import TensorType  # noqa: F401

from transport_processes import results_dir
from transport_processes.data.base_databatch import TransportProcessBatch


@dataclass
class GeminiVisualizationCheckpoint(pl.Callback):
    """Full-NP-Gemini visualization logic for 1D regression plots."""

    context_points: int = 10
    model_label: str = "gemini"
    _last_logged_epoch: int | None = field(default=None, init=False, repr=False)

    def _resolve_plot_dir(self, trainer) -> Path:
        root_dir = getattr(trainer, "default_root_dir", None)
        if root_dir:
            plot_root = Path(root_dir) / "training_images"
        else:
            plot_root = Path(results_dir) / "training_images"
        plot_root.mkdir(parents=True, exist_ok=True)
        return plot_root

    def _log_figure(self, trainer, fig, image_path: str) -> None:
        loggers = getattr(trainer, "loggers", None)
        if loggers is None:
            loggers = [trainer.logger]
        for logger in loggers:
            if logger is None:
                continue
            experiment = getattr(logger, "experiment", None)
            if experiment is None:
                continue
            if fig is not None and hasattr(experiment, "add_figure"):
                experiment.add_figure(
                    "val/predictions",
                    fig,
                    global_step=trainer.current_epoch,
                )
            elif hasattr(experiment, "log_image"):
                experiment.log_image(
                    image_path,
                    name="val/predictions",
                    step=trainer.current_epoch,
                )

    def _image_logging_period(self, trainer_cfg, trainer) -> int:
        max_epochs = getattr(trainer, "max_epochs", None)
        if max_epochs is None or max_epochs <= 0:
            max_epochs = getattr(trainer_cfg, "max_epochs", 0)
        return max(1, round(max_epochs * getattr(trainer_cfg, "log_images_every_pct", 0.0)))

    def _should_log_images(self, trainer_cfg, trainer) -> bool:
        enabled = getattr(trainer_cfg, "log_images_every_pct", 0.0) > 0
        if not enabled:
            return False
        period = self._image_logging_period(trainer_cfg, trainer)
        return (trainer.current_epoch % period) == 0

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        trainer_cfg = getattr(pl_module, "trainer_cfg", None)
        if trainer_cfg is None:
            return
        if not self._should_log_images(trainer_cfg, trainer):
            return
        if self._last_logged_epoch == trainer.current_epoch:
            return

        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is None:
            return
        try:
            val_loader = datamodule.val_dataloader()
        except Exception:
            return
        try:
            batch = next(iter(val_loader))
        except StopIteration:
            return

        if hasattr(batch, "to"):
            batch = batch.to(pl_module.device)
        elif isinstance(batch, (list, tuple)):
            batch = [t.to(pl_module.device) if hasattr(t, "to") else t for t in batch]

        output_root = self._resolve_plot_dir(trainer)
        model = getattr(pl_module, "model", pl_module)
        image_artifacts = self.log_images(
            model,
            batch,
            epoch=trainer.current_epoch,
            output_root=output_root,
        )
        for path, fig in image_artifacts:
            self._log_figure(trainer, fig, str(path))
            try:
                import matplotlib.pyplot as plt

                plt.close(fig)
            except Exception:
                pass
        self._last_logged_epoch = trainer.current_epoch

    def log_images(
        self,
        model,
        batch: TransportProcessBatch,
        *,
        epoch: int,
        output_root: Path | None = None,
    ) -> list[tuple[Path, object]]:
        if getattr(model, "x_dim", None) != 1 or getattr(model, "y_dim", None) != 1:
            return []

        if batch.target_output is None:
            return []

        import matplotlib.pyplot as plt

        device = batch.target_input.device
        idx = 0

        target_mask = batch.target_mask
        if target_mask is None:
            target_mask = torch.ones(batch.target_input.shape[:2], device=device, dtype=torch.bool)

        x_true = batch.target_input[idx][target_mask[idx]]
        y_true = batch.target_output[idx][target_mask[idx]]
        if x_true.numel() == 0:
            return []

        n_ctx = min(int(self.context_points), x_true.size(0))
        perm = torch.randperm(x_true.size(0), device=device)
        ctx_idx = perm[:n_ctx]

        x_c = x_true[ctx_idx].unsqueeze(0)
        y_c = y_true[ctx_idx].unsqueeze(0)

        x_min, x_max = getattr(model.exp_cfg.data, "gp_min_max", (-2.0, 2.0))
        n_grid = int(getattr(model.exp_cfg.data, "num_of_target_grid_inputs_for_logging", 100))
        x_plot = torch.linspace(x_min, x_max, n_grid, device=device).view(1, n_grid, 1)

        with torch.no_grad():
            mu, sigma = model.predict_from_context(x_c, y_c, x_plot)

        fig = plt.figure(figsize=(10, 6))
        plt.scatter(x_c[0].detach().cpu().numpy(), y_c[0].detach().cpu().numpy(), c="black", s=60)
        plt.plot(
            x_plot[0, :, 0].detach().cpu().numpy(),
            mu[0, :, 0].detach().cpu().numpy(),
            "b-",
        )
        sigma_np = sigma[0, :, 0].detach().cpu().numpy()
        mu_np = mu[0, :, 0].detach().cpu().numpy()
        x_plot_np = x_plot[0, :, 0].detach().cpu().numpy()
        plt.fill_between(
            x_plot_np,
            mu_np - 2 * sigma_np,
            mu_np + 2 * sigma_np,
            color="b",
            alpha=0.2,
        )
        plt.scatter(
            x_true.detach().cpu().numpy(),
            y_true.detach().cpu().numpy(),
            c="gray",
            alpha=0.3,
            s=10,
        )
        plt.title(f"Epoch {epoch}")
        plt.ylim(-3, 3)

        output_dir = Path(output_root or Path("training_images")) / self.model_label
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = output_dir / f"epoch_{epoch:03d}.png"
        plt.savefig(image_path)
        return [(image_path, fig)]
