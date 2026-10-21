"""Create report figures from saved experiment logs."""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from warehouse.analysis import read_csv, rolling, validate_greedy_checks
from warehouse.provenance import recorded_run_folders


def grid(ax, cfg):
    env = cfg["environment"]
    a = np.zeros((env["height"], env["width"]))
    for x, y in env["shelves"]:
        a[y, x] = 1
    ax.imshow(a, cmap="Greys", vmin=0, vmax=1, alpha=.65)
    ax.set_xticks([0, 2, 4, 6, 8]); ax.set_yticks([0, 2, 4, 6, 8])
    ax.set_xticks(np.arange(-.5, env["width"]), minor=True)
    ax.set_yticks(np.arange(-.5, env["height"]), minor=True)
    ax.grid(which="minor", alpha=.25)
    ax.tick_params(which="minor", length=0)
    for name, text in [("start", "S"), ("pickup", "P"), ("delivery", "D")]:
        ax.text(*env[name], text, weight="bold", ha="center", va="center", fontsize=10,
                bbox={"facecolor":"white", "alpha":.9, "edgecolor":"none", "pad":1})
    ax.set_xlabel("Column"); ax.set_ylabel("Row")


def export():
    for dataset in ("main", "adaptation"):
        results = ROOT / "results" / dataset
        cfg = json.loads((results / "config.json").read_text())
        for method in ("q_learning", "dqn"):
            folders = recorded_run_folders(results, method, cfg["training"]["seeds"])
            for folder, seed in zip(folders, cfg["training"]["seeds"]):
                validate_greedy_checks(read_csv(folder / "greedy_checks.csv"), cfg, seed,
                                       read_csv(folder / "training.csv"))
    out = ROOT / "output/figures";out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8})
    main = ROOT / "results/main"
    main_cfg = json.loads((main / "config.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(7, 5), layout="constrained")
    for method, label, color in [("q_learning", "Q-Learning", "#2369a2"), ("dqn", "DQN", "#d57425")]:
        folders = recorded_run_folders(main, method, main_cfg["training"]["seeds"])
        logs = [read_csv(p / "training.csv") for p in folders]
        checks = [read_csv(p / "greedy_checks.csv") for p in folders]
        for ax, key in zip(axes[0], ("return", "success")):
            ys = np.array([rolling([float(r[key]) for r in log]) for log in logs])
            x = np.arange(1, ys.shape[1]+1);mean=ys.mean(0);sd=ys.std(0,ddof=1)
            low, high = mean-sd, mean+sd
            if key == "success":low,high=np.clip(low,0,1),np.clip(high,0,1)
            ax.plot(x, mean, color=color, label=label, linewidth=1.4)
            ax.fill_between(x,low,high,color=color,alpha=.13)
        x = [int(r["episode"]) for r in checks[0]]
        steps = np.array([[int(r["steps"]) for r in c] for c in checks])
        axes[1,0].plot(x,np.median(steps,0),color=color,label=label,linewidth=1.3)
        axes[1,0].fill_between(x,steps.min(0),steps.max(0),color=color,alpha=.13)
        success=np.array([[int(r["success"]) for r in c] for c in checks])
        axes[1,1].plot(x,success.mean(0),color=color,label=label,linewidth=1.3)
    axes[1,0].axhline(18,color="black",linestyle="--",linewidth=1,label="BFS 18")
    for ax,title,ylabel in zip(axes.flat,["Training return","Training delivery success","Greedy episode length","Greedy delivery success"],
                               ["Undiscounted return","Delivery fraction","Moves (failures capped at 200)","Fraction of trained policies"]):
        ax.set_title(title);ax.set_xlabel("Training episode");ax.set_ylabel(ylabel);ax.grid(alpha=.2)
        ax.set_xticks([0,1000,2000,3000]);ax.legend(loc="best",framealpha=.9)
    for ax in axes[:,1]:ax.set_ylim(-.03,1.03)
    fig.savefig(out / "main_learning.png",dpi=240);plt.close(fig)
    cfg=json.loads((main / "config.json").read_text())
    fig,axes=plt.subplots(1,2,figsize=(7,3.8),layout="constrained")
    for ax,method,title in zip(axes,["q_learning","dqn"],["Q-Learning: all 10 final policies","DQN: selected final policies"]):
        grid(ax,cfg);ax.set_title(title)
        records=[json.loads(p.read_text()) for p in sorted((main / method).glob("seed_*/result.json"))]
        if method=="dqn":records=[r for r in records if r["seed"] in [0,1,3,7]]
        for i,r in enumerate(records):
            a=np.array(r["route"],float)+(i-(len(records)-1)/2)*.025
            line,=ax.plot(a[:,0],a[:,1],linewidth=1.2,alpha=.8,
                         label=f"seed {r['seed']}: {r['steps']}{' failed' if not r['success'] else ''}")
            if not r["success"]:ax.scatter(a[-1,0],a[-1,1],s=35,marker="x",color=line.get_color())
        ax.legend(loc="upper center",bbox_to_anchor=(.5,-.16),ncol=2,fontsize=7,framealpha=.9)
    fig.savefig(out / "main_routes.png",dpi=240);plt.close(fig)
    adaptation=ROOT / "results/adaptation"
    cfg=json.loads((adaptation / "config.json").read_text())
    fig,axes=plt.subplots(1,2,figsize=(7,3.3),layout="constrained")
    grid(axes[0],cfg);axes[0].set_title("Changed layout and adapted routes")
    for x,y in [(3,5),(4,6)]:
        axes[0].add_patch(Rectangle((x-.5,y-.5),1,1,facecolor="#d75b52",edgecolor="white",alpha=.9))
    for i,(method,label,color) in enumerate([("q_learning","Q-Learning","#2369a2"),("dqn","DQN","#d57425")]):
        r=json.loads((adaptation / method / "seed_0/result.json").read_text())
        a=np.array(r["route"],float)+(i-.5)*.08
        axes[0].plot(a[:,0],a[:,1],color=color,label=f"{label} seed 0 (20)",linewidth=1.4)
        checks=[read_csv(p / "greedy_checks.csv") for p in
                recorded_run_folders(adaptation, method, cfg["training"]["seeds"])]
        xs=[0]+[int(r["episode"]) for r in checks[0]]
        ys=np.r_[0,np.mean([[int(r["success"]) for r in c] for c in checks],axis=0)]
        axes[1].plot(xs,ys,color=color,label=label,linewidth=1.5)
    axes[0].legend(loc="upper center",bbox_to_anchor=(.5,-.17),fontsize=7)
    axes[1].set_title("Delivery after further training")
    axes[1].set_xlabel("Additional training episode")
    axes[1].set_ylabel("Fraction of 10 policies delivering")
    axes[1].set_ylim(-.03,1.03);axes[1].set_xticks([0,250,500,750,1000]);axes[1].grid(alpha=.2);axes[1].legend(loc="lower right")
    fig.savefig(out / "adaptation.png",dpi=240);plt.close(fig)
    print("Exported three report figures:",out)


if __name__ == "__main__":export()
