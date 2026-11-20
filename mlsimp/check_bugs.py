"""Shows three of MLSimp's code issues by calling the authors' own functions."""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

MLSIMP = Path(__file__).resolve().parents[1] / "third_party" / "MLSimp"


def index_offset(torch, params):
    from Model.GraphSimp import GAT
    from Utils.dataset import DiffuSimpcollate
    torch.manual_seed(0)
    gat = GAT(params["GAT"])
    z = torch.nn.functional.normalize(torch.randn(30, 32), dim=1)
    unit = torch.nn.functional.normalize(z, dim=1)
    neighbor = torch.topk(unit @ unit.T, 11, dim=-1).indices[:, 1:].T
    returned = gat.simp_trajs(z, neighbor, rate=0.2).numpy()

    align = sum((z - z[neighbor[i]]).norm(dim=1).pow(2) for i in range(neighbor.size(0))) / neighbor.size(0)
    uniform = torch.cdist(z, z).pow(2).mul(-2).exp().mean(dim=1).log()
    important = torch.softmax(align * uniform, dim=0)
    interior = np.arange(1, 29)
    best = np.sort(interior[np.argsort(-important[1:-1].numpy(), kind="stable")[:len(returned)]])

    traj = [[float(i), 0.0, 0.0] for i in range(30)]
    batch = DiffuSimpcollate([(traj, torch.tensor(returned))], 1.0, 1.0, 0.0, 0.0, "cpu")
    used = batch[4][0, 1:-1].numpy()

    print("3. Index offset (Model/GraphSimp.py simp_trajs, Utils/dataset.py DiffuSimpcollate)")
    print(f"   points with the highest importance: {best.tolist()}")
    print(f"   indices returned by simp_trajs:      {returned.tolist()}")
    print(f"   positions used as labels for Diff-TS: {used.tolist()}")
    print(f"   off by one: {bool(np.array_equal(returned + 1, best) and np.array_equal(used, returned))}")


def star_graph():
    from Utils.graph_utils import graph_constructor

    class Cells:
        grid_size = 10

        def gps2idx(self, lon, lat):
            return int(lon)

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "TrajData" / "Toy_out"
        folder.mkdir(parents=True)
        (folder / "0").write_text("".join(f"0.0 {cell}.5 {i}\n" for i, cell in enumerate([4, 5, 6, 7, 8])))
        here = os.getcwd()
        os.chdir(tmp)
        try:
            graph = graph_constructor(Cells(), "Toy")
        finally:
            os.chdir(here)
    print("4. Star graph (Utils/graph_utils.py graph_constructor)")
    print("   trajectory visits cells 4 -> 5 -> 6 -> 7 -> 8")
    print(f"   edges built:    {sorted(tuple(sorted(e)) for e in graph.edges())}")
    print("   edges expected: [(4, 5), (5, 6), (6, 7), (7, 8)]")


def sampling_last_step(torch, params):
    from DiffTeacher.model_util import create_model_and_diffusion
    d = params["Diff"]
    model, diffusion = create_model_and_diffusion(
        d["in_dim"], d["encoder_n_head"], d["encoder_hidden_dim"], d["encoder_n_layer"],
        d["in_channels"], d["hidden_channels"], d["out_channels"], d["n_head"], d["n_layer"],
        d["trans_hidden_channels"], d["attn_dropout"], d["dropout"], d["learn_sigma"],
        d["sigma_small"], d["diffusion_steps"], d["noise_schedule"], d["timestep_respacing"],
        d["use_kl"], d["predict_xstart"], d["rescale_timesteps"], d["rescale_learned_sigmas"],
        d["training_mode"])
    model.eval()
    torch.manual_seed(1)
    traj = torch.rand(2, 12, 4)
    mask = torch.zeros(2, 12, dtype=torch.bool)
    batch = (traj, mask, None, None, None, None)
    steps, amplify = 50, 5

    def run(seed):
        torch.manual_seed(seed)
        return diffusion.p_sample_loop(model, model_kwargs={"diffusion_steps": steps, "batch": batch,
                                                            "amplify_len": amplify},
                                       device="cpu", clip_denoised=False, top_p=-1)["sample"]

    with torch.no_grad():
        full = run(2)
        torch.manual_seed(2)
        start = model.get_embeds(traj, mask)
        x = torch.randn(2, 12 + amplify, start.size(2))
        x[:, :12] = start
        pad = torch.cat([mask, torch.zeros(2, amplify)], dim=-1)
        wrapped = diffusion._wrap_model(model)
        last = diffusion.p_sample(wrapped, x.clone(), pad, torch.zeros(2, dtype=torch.long),
                                  clip_denoised=False)["sample"]
        last[:, :12] = start
        iterative = x.clone()
        for i in reversed(range(steps)):
            iterative = diffusion.p_sample(wrapped, iterative, pad, torch.full((2,), i, dtype=torch.long),
                                           clip_denoised=False)["sample"]
            iterative[:, :12] = start
    print("5. Sampling uses only the last step (DiffTeacher/GaussianDiffusion.py p_sample_loop_progressive)")
    print(f"   {steps} steps vs only the step t=0, max difference: {(full - last).abs().max().item():.2e}")
    print(f"   {steps} steps that feed back, max difference:      {(full - iterative).abs().max().item():.2e}")


def main():
    os.chdir(MLSIMP)
    sys.path.insert(0, str(MLSIMP))
    import torch
    import yaml
    params = yaml.safe_load(open("Setting.yaml"))["Geolife"]
    index_offset(torch, params)
    star_graph()
    sampling_last_step(torch, params)


if __name__ == "__main__":
    main()
