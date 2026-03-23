import os, json, csv, time
import matplotlib.pyplot as plt

class ExperimentTracker:
    """
    NLP04 버전 계승 + NLP05용 수정
    - cfg 딕셔너리 주입 방식 유지
    - SFT / RM / PPO 단계별 실험 기록 지원
    """
    def __init__(self, cfg: dict, stage: str, results_dir: str):
        # stage: "SFT" | "RM" | "PPO"
        self.cfg = cfg
        self.stage = stage
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        self.log_path  = os.path.join(results_dir, f"{stage}_experiments_log.json")
        self.csv_path  = os.path.join(results_dir, f"{stage}_experiments_log.csv")
        self.plot_dir  = os.path.join(results_dir, f"{stage}_plots")
        os.makedirs(self.plot_dir, exist_ok=True)

        self.epoch_history = []   # {"epoch", "train_loss", "val_loss"}
        self.start_time    = time.time()
        self.all_records   = self._load_all()

    def _load_all(self):
        if os.path.exists(self.log_path):
            with open(self.log_path) as f:
                return json.load(f)
        return []

    def _next_id(self):
        return len(self.all_records) + 1

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float = None):
        entry = {"epoch": epoch, "train_loss": train_loss}
        if val_loss is not None:
            entry["val_loss"] = val_loss
        self.epoch_history.append(entry)
        print(f"[{self.stage}] Epoch {epoch:02d} | train_loss={train_loss:.4f}"
              + (f" | val_loss={val_loss:.4f}" if val_loss else ""))

    def save(self, best_epoch: int):
        duration = int(time.time() - self.start_time)
        record = {
            "id":         self._next_id(),
            "stage":      self.stage,
            "best_epoch": best_epoch,
            "duration_s": duration,
            "cfg":        self.cfg,
            "history":    self.epoch_history,
        }
        self.all_records.append(record)

        # JSON 저장
        with open(self.log_path, "w") as f:
            json.dump(self.all_records, f, ensure_ascii=False, indent=2)

        # CSV 저장 (플랫 형태)
        flat = {"id": record["id"], "stage": record["stage"],
                "best_epoch": record["best_epoch"], "duration_s": record["duration_s"]}
        flat.update({f"cfg_{k}": v for k, v in self.cfg.items()
                     if not isinstance(v, dict)})
        write_header = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=flat.keys())
            if write_header:
                w.writeheader()
            w.writerow(flat)

        self._generate_plots()
        self._print_summary(record)

    def _generate_plots(self):
        if not self.epoch_history:
            return
        epochs     = [h["epoch"]      for h in self.epoch_history]
        train_loss = [h["train_loss"] for h in self.epoch_history]
        val_loss   = [h.get("val_loss") for h in self.epoch_history]

        plt.figure(figsize=(8, 4))
        plt.plot(epochs, train_loss, label="train_loss")
        if any(v is not None for v in val_loss):
            plt.plot(epochs, val_loss, label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{self.stage} Training Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"exp{self._next_id()-1}_loss.png"))
        plt.close()

    def _print_summary(self, record):
        print(f"\n{'='*50}")
        print(f"[{self.stage}] Experiment #{record['id']} 완료")
        print(f"  Best epoch : {record['best_epoch']}")
        print(f"  Duration   : {record['duration_s']}s")
        if self.epoch_history:
            best = min(self.epoch_history, key=lambda x: x["train_loss"])
            print(f"  Best train loss : {best['train_loss']:.4f} (epoch {best['epoch']})")
        print(f"{'='*50}\n")
