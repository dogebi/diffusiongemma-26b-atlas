#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build diffusiongemma-26b-atlas/index.html  —  skill: tensor-atlas-v4-retarget.

src.html is the nisten "Tensor Atlas v4" engine (LLMViz-DeepSeek-V4.1-Flash, MIT (c) 2026
netsin) kept whole for the shell: topbar, view tabs, instanced WebGL2 renderer + Canvas2D
fallback, transport, inspector, ledger, safetensors audit, modals, CSS. Only the model half is
replaced: the data region (`const CFG` .. the line before `class CanvasRenderer`) plus the panel
copy that names the original model.

Subject: google/diffusiongemma-26B-A4B-it — block-diffusion Gemma 4 MoE. 30 blocks, hidden 2816,
128 experts top-8 beside an always-on 2112-wide dense MLP, 5 global + 25 sliding-window
attention blocks, a 27-block vision tower, a self-conditioning module for the diffusion decoder,
one tied 262144 x 2816 embedding, a 256-token diffusion canvas.

Bytes: three measured repositories, per-tensor sizes read from the safetensors headers over HTTP
Range (no weights downloaded), per-expert tensors folded into the fused banks and scale tensors
folded into their owner:
  BF16  google/diffusiongemma-26B-A4B-it                11 shards  51.6476 GB
  FP8   RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic   1 shard    27.1977 GB
  NVFP4 nvidia/diffusiongemma-26B-A4B-it-NVFP4           2 shards   18.8181 GB
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

DIR = Path(__file__).resolve().parent
SRC, OUT = DIR / "src.html", DIR / "index.html"
BF_JSON, FP8_JSON, NV_JSON = DIR / "hf-bf16.json", DIR / "hf-fp8.json", DIR / "hf-nvfp4.json"
FP8_CFG, NV_CFG = DIR / "hf-config-fp8.json", DIR / "hf-config-nvfp4.json"

SCALE_SUFFIX = re.compile(r"\.(weight_scale_2|weight_scale|input_scale|output_scale)$")
EXPERT = re.compile(r"^model\.decoder\.layers\.(\d+)\.experts\.\d+\.(gate_proj|up_proj|down_proj)(\.weight)?$")


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def make_folder(bfset: set[str]):
    def fold_expert(n: str) -> str | None:
        m = EXPERT.match(n)
        if not m:
            return None
        tail = "gate_up_proj" if m.group(2) in ("gate_proj", "up_proj") else "down_proj"
        return f"model.decoder.layers.{m.group(1)}.experts.{tail}"

    def folder(name: str) -> str | None:
        stripped = SCALE_SUFFIX.sub("", name)
        for c in (name, stripped, stripped + ".weight"):
            if c in bfset:
                return c
            f = fold_expert(c)
            if f and f in bfset:
                return f
        return None
    return folder


def collect(path: Path, bfset: set[str]) -> dict:
    folder = make_folder(bfset)
    out: dict[str, dict] = {}
    unmapped: list[str] = []
    for n, t in load(path)["tensors"].items():
        lg = folder(n)
        if lg is None:
            unmapped.append(n)
            continue
        rec = out.setdefault(lg, {"bytes": 0, "dims": t["shape"], "parts": 0})
        rec["bytes"] += t["bytes"]
        rec["parts"] += 1
    if unmapped:
        raise SystemExit(f"{path.name}: {len(unmapped)} tensors map to nothing, e.g. {unmapped[:4]}")
    return out


def templates() -> dict:
    bfset = set(load(BF_JSON)["tensors"])
    bf, fp8, nv = collect(BF_JSON, bfset), collect(FP8_JSON, bfset), collect(NV_JSON, bfset)
    for label, src in (("fp8", fp8), ("nvfp4", nv)):
        short = sorted(set(src) - set(bf))
        if short:
            raise SystemExit(f"{label}: logical names with no BF16 counterpart: {short[:3]}")
    both = {n: {"dims": bf[n]["dims"], "b16": bf[n]["bytes"],
                "b8": fp8[n]["bytes"], "b4": nv[n]["bytes"]} for n in bf}

    def layer_of(i: int) -> dict:
        pre = f"model.decoder.layers.{i}."
        return {n[len(pre):]: v for n, v in both.items() if n.startswith(pre)}

    sigs: dict[str, dict] = {}
    for i in range(30):
        got = layer_of(i)
        if not got:
            raise SystemExit(f"layer {i} has no tensors")
        key = json.dumps({k: v["dims"] for k, v in sorted(got.items())})
        probe = {k: (v["b16"], v["b8"], v["b4"]) for k, v in got.items()}
        if key in sigs:
            if probe != sigs[key]["probe"]:
                raise SystemExit(f"layer {i} differs byte-wise from its archetype")
            sigs[key]["indices"].append(i)
        else:
            sigs[key] = {"indices": [i], "items": got, "probe": probe}
    arch: dict[str, dict] = {}
    for s in sigs.values():
        dom = "full" if s["items"]["self_attn.q_proj.weight"]["dims"][0] == 8192 else "slide"
        if dom in arch:
            raise SystemExit(f"two signatures claim '{dom}'")
        arch[f"lm_{dom}"] = s

    vis_layers: dict[str, list[int]] = {}
    for n in both:
        m = re.match(r"^model\.encoder\.vision_tower\.encoder\.layers\.(\d+)\.(.+)$", n)
        if m:
            vis_layers.setdefault(m.group(2), []).append(int(m.group(1)))
    if not vis_layers or {len(v) for v in vis_layers.values()} != {27}:
        raise SystemExit(f"vision blocks are not uniform: { {k: len(v) for k, v in vis_layers.items()} }")
    vis = {}
    for suffix, idx in vis_layers.items():
        v = both[f"model.encoder.vision_tower.encoder.layers.{idx[0]}.{suffix}"]
        vis[f"model.encoder.vision_tower.encoder.layers.*.{suffix}"] = {
            "dims": v["dims"], "b16": v["b16"] * 27, "b8": v["b8"] * 27,
            "b4": v["b4"] * 27, "count": 27}
    io = {n: dict(v, count=1) for n, v in both.items()
          if not n.startswith("model.decoder.layers.")
          and not re.match(r"^model\.encoder\.vision_tower\.encoder\.layers\.\d+\.", n)}
    return {"arch": arch, "vis": vis, "io": io}


CAT = [(r"experts", "expert"), (r"router", "router"), (r"self_attn", "attn"),
       (r"mlp\.", "shared"), (r"norm|_scale$|layer_scalar|std_", "norm")]


def cat_of(name: str) -> str:
    if "vision_tower" in name or "embed_vision" in name:
        return "vision"
    if "self_conditioning" in name:
        return "selfcond"
    if "embed_tokens" in name:
        return "vocab"
    if name.endswith("decoder.norm.weight"):
        return "head"
    for pat, c in CAT:
        if re.search(pat, name):
            return c
    return "norm"


NOTE = {
    "experts.gate_up_proj": "128 experts x (gate + up), fused into one bank",
    "experts.down_proj": "128 experts x (704 -> 2816), fused into one bank",
    "router.proj.weight": "router: 128 logits per token",
    "router.per_expert_scale": "per-expert routing scale",
    "router.scale": "pre-router scale",
    "mlp.gate_proj.weight": "shared dense MLP, gate",
    "mlp.up_proj.weight": "shared dense MLP, up",
    "mlp.down_proj.weight": "shared dense MLP, down",
    "self_attn.q_proj.weight": "Q projection",
    "self_attn.k_proj.weight": "K projection (GQA)",
    "self_attn.v_proj.weight": "V projection (GQA)",
    "self_attn.o_proj.weight": "attention output",
    "self_attn.q_norm.weight": "per-head Q norm",
    "self_attn.k_norm.weight": "per-head K norm",
    "layer_scalar": "per-layer output scale",
}
VISION_NOTE = {
    "attn.q_proj.linear.weight": "ViT attention, q",
    "attn.k_proj.linear.weight": "ViT attention, k",
    "attn.v_proj.linear.weight": "ViT attention, v",
    "attn.o_proj.linear.weight": "ViT attention, out",
    "mlp.gate_proj.linear.weight": "ViT MLP, gate",
    "mlp.up_proj.linear.weight": "ViT MLP, up",
    "mlp.down_proj.linear.weight": "ViT MLP, down",
}


def js_array(var: str, items: list[tuple[str, dict]], comment: str, vision: bool = False) -> str:
    lines = [f"const {var}=[", f"  /* {comment} */"]
    for name, v in items:
        note = ""
        if vision:
            note = VISION_NOTE.get(".".join(name.split(".")[-4:-1]), "")
        elif name in NOTE:
            note = NOTE[name]
        elif name.startswith("model.decoder.layers."):
            note = NOTE.get(name.split(".", 3)[-1], "")
        label = name + (f"  ({note})" if note else "")
        count = v.get("count", 1)
        extra = f",{count}" if count > 1 else ""
        lines.append(f'  W({json.dumps(label)},{json.dumps(v["dims"])},{json.dumps(cat_of(name))},'
                     f'"",{{bf16:{v["b16"]},fp8:{v["b8"]},nvfp4:{v["b4"]}}}{extra}),')
    lines.append("];")
    return "\n".join(lines)


DATA_HEAD = r"""// ---------- data.js ----------
/* DiffusionGemma 26B A4B — Tensor Atlas. Model facts come from the published config.json of
   google/diffusiongemma-26B-A4B-it. Every byte is audited against the safetensors headers of
   three published repositories (BF16, FP8-dynamic, NVFP4) and summed per tensor, with per-expert
   tensors folded into the fused banks and scale tensors into their owner. */
const CFG = {
 text_config:{
  model_type:'diffusion_gemma_text',num_hidden_layers:30,hidden_size:2816,
  intermediate_size:2112,moe_intermediate_size:704,num_experts:128,top_k_experts:8,
  num_attention_heads:16,num_key_value_heads:8,num_global_key_value_heads:2,
  head_dim:256,global_head_dim:512,rms_norm_eps:1e-6,
  vocab_size:262144,max_position_embeddings:262144,sliding_window:1024,
  tie_word_embeddings:true,final_logit_softcapping:30.0,use_bidirectional_attention:'vision',
  layer_types:["sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention"],
  kv_source_layer_ids:[5,11,17,23,29],index_source_layer_ids:[],
  rope_parameters:{sliding_attention:{rope_theta:10000.0,rope_type:'default'},
                   full_attention:{rope_theta:1000000.0,rope_type:'proportional',partial_rotary_factor:0.25}},
  bos_token_id:2,eos_token_id:1,pad_token_id:0
 },
 image_token_id:258880,canvas_length:256,
 sampling:{max_steps:48,temperature:'0.8 linear down to 0.4',entropy_bound:0.1,stop_entropy:0.005},
 vision_config:{model_type:'gemma4_vision',num_hidden_layers:27,hidden_size:1152,
  num_attention_heads:16,num_key_value_heads:16,head_dim:72,
  intermediate_size:4304,patch_size:16,default_output_length:280}
};
const CT=CFG.text_config, VC=CFG.vision_config;
"""

DATA_TAIL = r"""const COL={blue:'#638bff',enc:'#5ca7ff',dec:'#55d7c1',engram:'#b99bff',expert:'#76b9ff',shared:'#d6e99c',attn:'#55ddd0',router:'#f4b765',mhc:'#e3a8dc',vision:'#80ceea',head:'#c4d0ff',full:'#efbc71',reindex:'#b798ff',reuse:'#709bbd',swa:'#758090',norm:'#a7b5cc',muted:'#8390a6'};
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const lerp=(a,b,t)=>a+(b-a)*t;
const smooth=x=>x*x*(3-2*x);
const num=x=>Math.round(x).toLocaleString('en-US');
function fmtP(p){return p>=1e12?(p/1e12).toFixed(3)+'T':p>=1e9?(p/1e9).toFixed(2)+'B':p>=1e6?(p/1e6).toFixed(2)+'M':p>=1e3?(p/1e3).toFixed(1)+'K':num(p);}
function bytes(n,binary=false){let b=binary?1024:1000,u=binary?['B','KiB','MiB','GiB','TiB']:['B','KB','MB','GB','TB'],k=0;while(n>=b&&k<4){n/=b;k++;}return (k===0?num(n):n.toFixed(n>=100?1:2))+' '+u[k];}
const SOURCE_BASE='https://huggingface.co/google/diffusiongemma-26B-A4B-it';
const SOURCES=[
 ['Model card',SOURCE_BASE,'Block-diffusion Gemma 4 MoE: 25.2B total parameters, 3.8B active, 30 blocks, 8 of 128 experts per token, a 256-token canvas and 256K context. The benchmark table in this atlas is copied from that card.'],
 ['Released config',SOURCE_BASE+'/blob/main/config.json','Text and vision configs embedded field for field: layer_types (5 full / 25 sliding), 128 experts top-8, global_head_dim 512, canvas_length 256, sliding_window 1024, partial_rotary_factor 0.25 on the global blocks.'],
 ['BF16 checkpoint',SOURCE_BASE+'/tree/main','11 shards, 1,047 logical tensors, 51.6476 GB. Every byte on this page was read from these shard headers over HTTP Range; no weight data was downloaded.'],
 ['FP8-dynamic build','https://huggingface.co/RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic','compressed-tensors W8A8 in one 27.1977 GB shard: e4m3 weights with per-channel scales and per-token dynamic activation scales. Its 255-entry ignore list keeps the routers, layer_scalar tensors and every RMSNorm in BF16.'],
 ['NVFP4 build','https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4','modelopt NVFP4, 18.8181 GB in two shards. The ignore list is the honest part: lm_head, embed_vision, *mlp*, *router*, *self_attn*, *self_conditioning* and *vision_tower* stay BF16 — only the fused expert banks were converted.'],
 ['Licence',SOURCE_BASE+'/blob/main/README.md','Apache-2.0 (Gemma 4 licence). Page engine MIT, (c) 2026 netsin. Metadata only: no weight data is bundled in this file.']
];
const MODE_INFO={
 bf16:{label:'BF16 checkpoint',short:'BF16',color:COL.enc,note:'google/diffusiongemma-26B-A4B-it / 11 shards / 2 bytes per parameter'},
 fp8:{label:'FP8-dynamic',short:'FP8',color:COL.dec,note:'RedHatAI / compressed-tensors W8A8 / channel scales / routers and norms stay BF16'},
 nvfp4:{label:'NVFP4 experts',short:'NVFP4',color:COL.expert,note:'nvidia/modelopt / block-16 e2m1 + e4m3 scales / experts only'}
};
const MODE_KEYS=['bf16','fp8','nvfp4'];
/* There is no KV compression in this checkpoint. The two behaviours are the 1024-token sliding
   window (25 blocks) and the five global blocks that see the whole context. */
function modeFor(i){return CT.layer_types[i]==='full_attention'?'full':'swa';}
function ownerFor(i){return modeFor(i)==='full'?i:null;}
function indexOwnerFor(i){return null;}
function W(name,shape,cat,note='',ex=null,count=1){
 return {name,shape,cat,note,count,
   p:shape.reduce((a,b)=>a*b,1)*count,
   format:ex&&ex.nvfp4&&ex.nvfp4<ex.bf16*0.75?'nvfp4':(ex&&ex.fp8<ex.bf16*0.9?'fp8':'bf16'),
   ex};
}
function wBytes(w,mode='bf16'){return w.ex?w.ex[mode]:2*w.p;}
function wFormat(w,mode){return {bf16:'BF16',fp8:'FP8 / channel',nvfp4:'NVFP4 / 16'}[mode];}
const sumP=ws=>ws.reduce((a,w)=>a+w.p,0);
const sumB=(ws,m)=>ws.reduce((a,w)=>a+wBytes(w,m),0);
@@TABLES@@
const FP8_CONFIG=@@FP8CFG@@;
const NV_CONFIG=@@NVCFG@@;
function layerWeights(i){return modeFor(i)==='full'?FULL_W:SLIDE_W;}
const LAYERS=Array.from({length:30},(_,i)=>({id:'L'+i,index:i,label:'Block '+String(i).padStart(2,'0'),
 part:i<15?'encoder':'decoder',mode:modeFor(i),owner:ownerFor(i),indexOwner:indexOwnerFor(i),
 ratio:1,ws:layerWeights(i)}));
const ENGRAM=[];                       /* this checkpoint has no memory tables */
const DRAFT=[];                        /* the canvas adds no parameters: the same blocks run twice */
const EMBED={id:'embed',label:'Token embedding (tied)',ws:EMBED_W};
const HEAD={id:'head',label:'Output norm',ws:HEAD_W};
const SELF={id:'selfcond',label:'Self-conditioning',ws:SELFCOND_W};
const VISION={id:'vision',label:'Vision tower · 27 blocks',ws:VISION_W};
const ALIGNER={id:'aligner',label:'Encoder interface',ws:ENC_W};
const MODULES=[EMBED,...LAYERS,HEAD,SELF,VISION,ALIGNER];
const ALL_W=MODULES.flatMap(m=>m.ws);
const TOTALS=Object.fromEntries(MODE_KEYS.map(m=>[m,sumB(ALL_W,m)]));
const TOTAL_P=sumP(ALL_W);
const MOE_EXPERT_P=(2*2816*704+704*2816);
const MOE_TOTAL_P=MOE_EXPERT_P*128*30;
const NV_DELTA=TOTALS.bf16-TOTALS.nvfp4;
const FP8_DELTA=TOTALS.bf16-TOTALS.fp8;
const KV_SLIDE_PER_TOKEN=2*8*256*2;
const KV_FULL_PER_TOKEN=2*2*512*2;
const KV_SLIDE_TOTAL=KV_SLIDE_PER_TOKEN*1024*25;
const CATEGORIES=[
 ['expert','Routed experts · 128 per block',COL.expert,LAYERS.flatMap(l=>l.ws.filter(w=>w.cat==='expert'))],
 ['attn','Attention',COL.attn,LAYERS.flatMap(l=>l.ws.filter(w=>w.cat==='attn'))],
 ['shared','Shared dense MLP',COL.shared,LAYERS.flatMap(l=>l.ws.filter(w=>w.cat==='shared'))],
 ['router','Routers and scales',COL.router,LAYERS.flatMap(l=>l.ws.filter(w=>w.cat==='router'))],
 ['vision','Vision tower + projector',COL.vision,[...VISION.ws,...ALIGNER.ws]],
 ['selfcond','Self-conditioning',COL.engram,SELF.ws],
 ['vocab','Tied embedding',COL.head,EMBED.ws],
 ['norm','Norms and layer scales',COL.norm,ALL_W.filter(w=>w.cat==='norm')]
];
const EXP={
 overview:{title:'Two passes, one set of blocks.',body:'DiffusionGemma is a Gemma 4 Mixture-of-Experts that writes text by discrete diffusion instead of one-token-at-a-time sampling. The checkpoint holds ONE set of 30 blocks: the encoder runs them causally over the prompt to build the KV cache, then the decoder runs them bidirectionally over a 256-token canvas, denoising it in parallel. Weights are shared; the two roles are not. 45.7 GB of the 51.6 GB checkpoint is the expert banks, which is why both quantized builds attack the same 88%.'},
 expert:{title:'128 experts, 8 per token, 88% of the file.',body:'Every block stores its experts fused: gate_up_proj [128, 1408, 2816] and down_proj [128, 2816, 704] — one tensor per bank, all 128 experts stacked inside it. That is 1015.0 MB + 507.5 MB per block, 45.7 GB over 30 blocks: 88.4% of the BF16 checkpoint. All 128 experts stay on disk for every token; the router picks 8 and mixes them by weight, beside an always-on dense MLP with the same width budget (2112 intermediate).'},
 full:{title:'The five global blocks.',body:'Blocks 5, 11, 17, 23 and 29 leave the sliding window and attend the whole context: 16 query heads over 2 global KV heads at head_dim 512, so k is 1024 wide and q/o are 8192 wide. These five store no separate v_proj — only q, k and o are in the checkpoint. Their rotary position covers a quarter of each head (partial_rotary_factor 0.25, theta 1e6), while the other twenty-five use theta 1e4.'},
 swa:{title:'The twenty-five sliding blocks.',body:'A 1024-token window: 16 query heads over 8 KV heads at head_dim 256, with per-head Q and K norms. Their KV cache is capped at 1024 positions, so a 256K conversation costs them nothing beyond the first kilotoken; only the five global blocks grow. Attention is under 4% of a sliding block — 1.52 GB of it is the fused expert banks at BF16.'},
 selfcond:{title:'Self-conditioning: 35.7 MB that make it a diffusion model.',body:'The decoder is not autoregressive. It denoises all 256 canvas positions at once and can feed its previous estimate back in, which is what this module does: a pre-norm plus gate/up/down projections (2816 -> 2112 -> 2816). The encoder pass never calls it, and its absence from the ignore lists of both quantized builds is part of why it stays BF16.'},
 vision:{title:'A 27-block vision tower and a 6.5 MB door.',body:'Images are patchified at 16x16 and run through 27 blocks of hidden 1152 — 16 heads at head_dim 72, a 4304-wide MLP — then projected by embed_vision [2816, 1152] into the language hidden size. The tower carries its own patch embedder, a [2, 10240, 1152] position table and two 1152-wide correction vectors. 1.1 GB of the checkpoint, BF16 in all three builds.'},
 bf16:{title:'The reference checkpoint.',body:'51.6476 GB in 11 shards: 45.7 GB of routed experts, 1.48 GB of tied embedding, 1.06 GB of attention, about 1.1 GB of vision tower, 0.36 GB of shared dense MLPs, 36 MB of self-conditioning, and the rest norms, routers and layer scales.'},
 fp8:{title:'FP8-dynamic: everything except the routers, norms and embedding.',body:'27.1977 GB in a single shard — 1.90x smaller. Every linear weight becomes e4m3 with a per-channel scale and activations get per-token dynamic scales, experts included. The 255-entry ignore list keeps the routers, layer_scalar tensors and RMSNorms in BF16, and the 1.48 GB embedding is untouched too.'},
 nvfp4:{title:'NVFP4: only the experts move.',body:'18.8181 GB in two shards — 2.74x smaller than BF16. The reason it is not smaller is the ignore list: lm_head, embed_vision, *mlp*, *router*, *self_attn*, *self_conditioning* and *vision_tower* all stay BF16. Only the fused expert banks are block-16 e2m1 with e4m3 scales, and each expert adds a per-tensor F32 scale. The tied embedding alone is 1.48 GB of these 18.8 GB.'},
 storage:{title:'Where 51.6 GB goes.',body:'88.4% is the routed experts. Flip the precision and watch which categories move: NVFP4 converts exactly one of them, FP8 converts everything except the embedding, the routers and the norms.'},
 cache:{title:'Two attention behaviours in one checkpoint.',body:'Twenty-five blocks see 1024 tokens; five see everything. At 256K context the sliding blocks cost what they cost at 1K, and only the five global blocks keep growing — k 1024 wide at BF16, with no separate v_proj in the checkpoint.'}
};
const BENCH=[['GPQA Diamond','Reasoning',[73.2,93.4,94.1,92.9,88.1,92.4,89.9,90.9]],['Terminal-Bench 2.1','Agentic',[null,89.1,88.8,88.3,88.2,87.9,82.7,90.6]],['Terminal-Bench 3.0','Agentic',[null,43.3,34.4,17.7,28.3,11.8,7.6,30]],['Terminal-Bench 4.0','Agentic',[null,51.8,39.9,12.6,37.9,12.4,7,31.2]],['DeepSWE v1.1','Agentic',[null,74,73,67.5,66.9,62.7,54.4,74.2]],['ProgramBench','Agentic',[null,37,23,17.5,19,15.5,null,20.3]],['NL2Repo-Bench','Agentic',[null,75.3,56.8,58,58,61.5,54.2,64]],['CyberGym','Agentic',[null,null,84.5,80,84.5,83.3,76.7,88.1]],['SEC-Bench Pro','Agentic',[null,null,74.3,null,null,56.4,30.9,62.8]],['ExploitGym','Agentic',[null,22.1,33.7,null,15,5.4,1.8,15.3]],['HLE with tools','Agentic',[null,63.6,null,59.8,62.5,60,51.5,63.9]],['AutomationBench','Agentic',[null,50.3,45.8,46.7,48.8,43.2,37.7,54.8]],['Agent\'s Last Exam','Agentic',[null,28.6,26.7,27.6,28.5,25.7,25.2,31.8]],['Chartography with tools','Visual',[null,84,79.9,68.1,null,null,null,78.9]],['BabyVision with tools','Visual',[null,94.1,88.9,85.7,null,null,null,89.6]],['ZeroBench-main (Pass@5)','Visual',[null,52,53,41,null,null,null,49]]];
const BENCH_MODELS=['DiffusionGemma 26B A4B (this atlas)','Opus-5.0','GPT-5.6 Sol','K3','GLM-5.3','DS V4 Pro','DS V4 Flash','DS V4.1 Flash'];
/* Publisher-reported leaderboard as published on the original atlas page (each model's
   own card). DiffusionGemma 26B A4B is not in that publisher set; its own card figure is shown first for the benchmarks its card publishes, and a blank cell means the card does not publish that benchmark. */

/* Publisher-reported leaderboard as published on the original atlas page (each model's
   own card). DiffusionGemma 26B A4B is not in that publisher set; its own card figure is shown first for the benchmarks its card publishes, and a blank cell means the card does not publish that benchmark. */

function pickExperts(seed,n=128,k=8){let x=(seed+1)*2654435761>>>0;const s=new Set;while(s.size<k){x=(Math.imul(x,1664525)+1013904223)>>>0;s.add(x%n);}return [...s].sort((a,b)=>a-b);}
const PHASES=[
 {name:'Encode',label:'Encoder',from:0,to:6,active:'prefill',color:COL.enc,caption:'Read the prompt, build the KV cache.',desc:'The 30 blocks run causally over the prompt. Every token routes to 8 of 128 experts; the 25 sliding blocks keep a bounded 1024-token window and the 5 global blocks keep everything. Images enter the same stream through the vision tower and embed_vision.'},
 {name:'Canvas',label:'Canvas',from:6,to:11,active:'256 positions',color:COL.dec,caption:'Start from noise: 256 masked positions.',desc:'The decoder does not generate one token at a time. All 256 canvas positions start masked and are denoised in parallel with bidirectional attention over the canvas, reaching the cached prompt through the same blocks.'},
 {name:'Denoise',label:'Denoise',from:11,to:17,active:'up to 48 steps',color:COL.expert,caption:'Iterate until the canvas is confident.',desc:'Every step re-predicts all positions. The sampler keeps the lowest-entropy tokens whose mutual-information bound stays under 0.1, renoises the rest, and stops early when the average canvas entropy falls below 0.005 and two consecutive steps agree. Self-conditioning feeds the previous estimate into the next step.'},
 {name:'Append',label:'Append',from:17,to:20,active:'encoder again',color:COL.shared,caption:'The accepted canvas joins the context.',desc:'A finished canvas is appended to the KV cache and the next pass begins — which is why one weight set serves both directions: the encoder stack and the decoder stack are the same 30 blocks.'},
 {name:'Repeat',label:'Repeat',from:20,to:22,active:'15-20 tok/pass',color:COL.full,caption:'Many tokens per forward pass.',desc:'Because a whole canvas is denoised at once, 15-20 tokens leave the model per forward pass, which is where the reported 1100+ tokens per second at low batch size comes from.'}
];
const TRAIN_PHASES=[
 {name:'Forward',from:0,to:8,active:'Read W',color:COL.enc,caption:'Corrupt a canvas, denoise it.',desc:'Conceptual training pass: canvas tokens are masked, the blocks predict them, routing is chosen per token. A teaching schematic, not a reproduction of Google training recipe.'},
 {name:'Loss',from:8,to:11,active:'Compare',color:COL.engram,caption:'Score the whole canvas at once.',desc:'Diffusion training scores every canvas position rather than one next token. Values here illustrate shape, not the real loss mixture.'},
 {name:'Backward',from:11,to:18,active:'Gradients',color:COL.mhc,caption:'Trace the error back.',desc:'Gradients follow the forward operations; optimizer state is not part of the weight solids on this page.'}
];
function phasesFor(mode){return mode==='training'?TRAIN_PHASES:PHASES;}
function phaseAt(t,mode='inference'){const ps=phasesFor(mode);return ps.find(p=>t>=p.from&&t<p.to)||ps[ps.length-1];}
const TC={q:'#8caaff',local:'#ffc477',kv:'#61dccb',index:'#f5a078',out:'#b2bfff',expert:'#66deb0',shared:'#eee081',router:'#f494be',mhc:'#c9a3fa',norm:'#90a9b5',engram:'#d7b57b',vision:'#62d5d0',head:'#a5b9eb',attn:'#55ddd0',selfcond:'#b99bff',vocab:'#a5b9eb'};
function tensorKind(w){return TC[w.cat]?w.cat:'head';}
function tensorColor(w){return TC[tensorKind(w)]||COL.attn;}
function tensorShort(w){return w.name.replace(/^model\.decoder\.layers\.\d+\./,'').replace(/^model\.encoder\.vision_tower\.encoder\.layers\.\*\./,'ViT.').replace(/^model\.encoder\./,'enc.').replace(/^model\.decoder\./,'').replace(/\.weight$/,'').replace(/\.linear$/,'');}
function weightParts(w,mode){const total=wBytes(w,mode);return {data:total,scales:0,aux:0,total};}
function displayWeights(m){return m.ws;}
function findWeight(m,name){return displayWeights(m).find(w=>w.name===name)||m.ws.find(w=>w.name===name);}
function orderWeights(m){const ws=displayWeights(m);const rank=w=>{let n=w.name;
 if(n.includes('experts.gate_up'))return 0; if(n.includes('experts.down'))return 1;
 if(n.includes('router.proj'))return 2; if(n.includes('router.per_expert'))return 3; if(n.endsWith('router.scale'))return 4;
 if(n.includes('mlp.gate'))return 5; if(n.includes('mlp.up'))return 6; if(n.includes('mlp.down'))return 7;
 if(n.includes('q_proj'))return 8; if(n.includes('q_norm'))return 9; if(n.includes('k_proj'))return 10;
 if(n.includes('k_norm'))return 11; if(n.includes('v_proj'))return 12; if(n.includes('o_proj'))return 13;
 if(n.includes('input_layernorm'))return 14; if(n.includes('post_attention'))return 15;
 if(n.includes('feedforward'))return 16; if(n.includes('layer_scalar'))return 17;
 if(n.includes('self_conditioning'))return 18; if(n.includes('embed_tokens'))return 0;
 if(n.includes('patch_embedder'))return 0; if(n.includes('std_'))return 1; if(n.includes('embed_vision'))return 2;
 if(n.includes('vision_tower.encoder'))return 3; return 19;};
 return [...ws].sort((a,b)=>rank(a)-rank(b));}
function isAttentionTensor(w){return w.cat==='attn';}
function tensorInfo(w,m){
 const p=fmtP(w.p),size=bytes(wBytes(w,'bf16'));
 let t='Stored tensor.',b='Two bytes per parameter in the BF16 checkpoint. The storage view shows what each quantized build does with it.';
 if(w.name.includes('experts.gate_up')){t='Fused expert gate and up bank.';b='One tensor holds all 128 experts: [128, 1408, 2816]. Inside each expert, gate (704 x 2816) and up (704 x 2816) sit side by side and the block computes silu(gate) x up. 1015.0 MB per block at BF16, 30.5 GB across the stack — the largest tensor class in the checkpoint.';}
 else if(w.name.includes('experts.down')){t='Fused expert down bank.';b='[128, 2816, 704]: each expert projects its 704-wide activation back to the 2816-dim residual stream. 507.5 MB per block at BF16, 15.2 GB across the stack.';}
 else if(w.name.includes('router.proj')){t='Router.';b='2816 -> 128 logits, one per expert. The top 8 fire and their outputs are mixed by the routing weights, with the always-on dense MLP beside them. 0.72 MB per block, and both quantized builds refuse to touch it: a misrouting router costs more than an 8-bit one.';}
 else if(w.name.includes('router.per_expert_scale')){t='Per-expert routing scale.';b='128 learned scalars multiplying each expert routing weight before the top-k selection.';}
 else if(w.name.endsWith('router.scale')){t='Pre-router scale.';b='2816 learned scales applied to the hidden state before routing — Gemma-style decoration of the router input.';}
 else if(w.name.includes('mlp.')){t='Shared dense MLP.';b='The always-on path beside the experts: 2816 -> 2112 -> 2816 with gelu-tanh. Every token passes through it, so it is the one feed-forward part with no routing and no sparsity. 11.9 MB per matrix, 35.7 MB per block.';}
 else if(w.name.includes('self_conditioning')){t='Self-conditioning projection.';b='2816 -> 2112 -> 2816 plus a pre-norm, applied to the decoder previous estimate before the next denoising step. 35.7 MB in total, unique to the diffusion decoder, BF16 in every build.';}
 else if(w.name.includes('q_proj')){t='Query projection.';b='The five global blocks project to 8192 (16 heads x 512); the twenty-five sliding blocks project to 4096 (16 heads x 256). Per-head Q norm follows in both.';}
 else if(w.name.includes('k_proj')){t='Key projection.';b='Global blocks: 2 KV heads x 512 = 1024 rows. Sliding blocks: 8 KV heads x 256 = 2048. Grouped-query attention, with per-head K norm.';}
 else if(w.name.includes('v_proj')){t='Value projection.';b='Present in the twenty-five sliding blocks only: 8 KV heads x 256 = 2048 rows. The five global blocks store no separate v_proj in this checkpoint.';}
 else if(w.name.includes('o_proj')){t='Attention output projection.';b='4096 -> 2816 in the sliding blocks, 8192 -> 2816 in the global ones.';}
 else if(w.name.includes('q_norm')||w.name.includes('k_norm')){t='Per-head norm.';b='256 scales in the sliding blocks, 512 in the global ones: RMSNorm applied per head to queries and keys before the dot product.';}
 else if(w.name.includes('layer_scalar')){t='Layer scalar.';b='A single learned scalar multiplying the block output. It appears once per block in the decoder stack and once more under model.encoder.language_model.layers — separate layer scales for the encoder pass over the same weights, which is the clearest evidence that one weight set serves both directions.';}
 else if(w.name.includes('embed_tokens')){t='Token embedding — and the unembedding too.';b='[262144, 2816]: 1.48 GB, the whole vocabulary, tied (tie_word_embeddings true), which is why the checkpoint has no lm_head tensor at all. Both quantized builds keep it BF16.';}
 else if(w.name.includes('patch_embedder')){t='Patch embedder and position table.';b='16x16 patches, plus a [2, 10240, 1152] learned position table — the vision tower keeps its own position signal, separate from the language rotary.';}
 else if(w.name.includes('embed_vision')){t='Vision-to-language projector.';b='[2816, 1152]: the 6.5 MB door between the 27-block vision tower and the language hidden size.';}
 else if(w.name.includes('std_bias')||w.name.includes('std_scale')){t='Vision output correction.';b='Two 1152-wide vectors rescaling the tower output before projection.';}
 else if(w.name.includes('vision_tower')){t='Vision tower tensor.';b='Part of the 27-block ViT: attention with 16 heads at head_dim 72, or the 4304-wide GELU-tanh MLP. Summed over 27 blocks; BF16 in all three builds.';}
 else if(w.name.includes('norm')){t='RMSNorm.';b='Layer normalisation on the residual stream (eps 1e-6). Kept BF16 by both quantized builds.';}
 return {title:t,body:b,size,p};}
function layerStory(m){
 if(!m.mode)return null;let t,body,detail;
 if(m.mode==='swa'){t='Sliding-window block';body='1024-token attention on 8 KV heads, then 8 of 128 experts plus the shared dense MLP.';detail='This is 25 of the 30 blocks. Its KV cache is capped at 1024 positions, so long context costs it nothing after the first kilotoken. The fused expert banks are 1.52 GB of this block at BF16; attention is under 4%.';}
 else {t='Global block';body='Full-context attention on 2 global KV heads at head_dim 512, then the same 8-of-128 expert bank.';detail='Five blocks — 5, 11, 17, 23, 29 — keep a KV cache that grows with the context. They are the only blocks whose attention shapes differ: q/o 8192 wide, k 1024, and no separate v_proj in the checkpoint.';}
 return {title:t,body,detail};}
"""


def build_data_js(t: dict) -> str:
    arch, io = t["arch"], t["io"]
    slide = sorted(arch["lm_slide"]["items"].items())
    full = sorted(arch["lm_full"]["items"].items())
    vis = sorted(t["vis"].items())

    def pick(pred):
        return sorted((n, v) for n, v in io.items() if pred(n))

    embed = pick(lambda n: "embed_tokens" in n)
    head = pick(lambda n: n.endswith("decoder.norm.weight"))
    selfcond = pick(lambda n: "self_conditioning" in n)
    enc = pick(lambda n: "embed_vision" in n or "vision_tower.patch_embedder" in n
               or "vision_tower.std_" in n or "encoder.language_model.layers" in n)
    missing = sorted(set(io) - {n for g in (embed, head, selfcond, enc) for n, _ in g})
    if missing:
        raise SystemExit(f"io tensors placed in no module: {missing[:5]}")
    tables = "\n".join([
        js_array("SLIDE_W", slide, "one sliding-window block: attention + router + 128 experts + dense MLP"),
        js_array("FULL_W", full, "one global block: q/k/o only (no v_proj), same router and expert banks"),
        js_array("EMBED_W", embed, "the tied embedding / unembedding"),
        js_array("HEAD_W", head, "final RMSNorm — the head itself is tied"),
        js_array("SELFCOND_W", selfcond, "diffusion self-conditioning"),
        js_array("VISION_W", vis, "one vision block x27 (bytes already summed into each entry)", vision=True),
        js_array("ENC_W", enc, "encoder side: patch embedder, position table, std vectors, vision projector, 30 layer scales"),
    ])
    js = (DATA_HEAD + DATA_TAIL)
    js = js.replace("@@TABLES@@", tables)
    js = js.replace("@@FP8CFG@@", json.dumps(load(FP8_CFG), ensure_ascii=False))
    js = js.replace("@@NVCFG@@", json.dumps(load(NV_CFG), ensure_ascii=False))
    return js


# ── panel rewrites: (anchor substring that must appear exactly once, replacement line) ─────────
R = lambda anchor, line: (anchor, line)  # noqa: E731

REWRITES: list[tuple[str, str]] = [
 # ---------- overview hero ----------
 R("ONE CHECKPOINT / ONE CONTINUOUS FORWARD PASS",
   '  <div class="intro-kicker">ONE WEIGHT SET / TWO PASSES IN OPPOSITE DIRECTIONS</div><h2 class="editorial">All 30 blocks.<br/><em>Twice per token.</em></h2><p class="lede">The encoder runs the blocks causally to read the prompt. The decoder runs the same blocks bidirectionally over a 256-token canvas and denoises it. Weights are shared; the roles are not.</p>'),
 R("Decode execution order",
   '  <div class="decode-path" aria-label="Generation order"><span>Embedding</span><i>\\u2192</i><span>Encoder pass<small>causal prefill</small></span><i>\\u2192</i><span>Canvas 256<small>masked start</small></span><i>\\u2192</i><span>Denoise<small>up to 48 steps</small></span><i>\\u2192</i><span>Append + repeat</span></div>'),
 R("Same block pattern, different weights",
   '  <${Section} title="Same blocks, two directions"><p class="fine">There is one set of 30 blocks, not two. In the encoder pass attention is causal and the KV cache is written; in the decoder pass attention is bidirectional over the 256 canvas positions and the cache is read. The layer scalars are stored twice — <b>model.decoder.layers.N.layer_scalar and model.encoder.language_model.layers.N.layer_scalar</b> — which is what makes the shared-weight claim visible in the checkpoint itself.</p><p class="fine">The split view separates the first fifteen blocks (whose outputs the encoder reads first) from the last fifteen; the sequence is still L00 through L29. The tower view shows the same objects in one column.</p><button class="text-link" onClick=${()=>this.setLayout(s.layout===\'tower\'?\'split\':\'tower\')}>${s.layout===\'tower\'?\'Separate the two role groups\':\'Merge into one numbered tower\'} <${Icon} name="arrow" size=${14}/></button><//>'),
 R("What actually changes at the boundary?",
   '  <${Section} title="What actually changes between the passes?"><p class="fine">The weights do not change — the attention mask and the input do. The encoder pass is causal, so each position sees only its predecessors, and its keys and values become the context the canvas reads. The decoder pass is bidirectional inside the 256-token canvas, so every masked position sees every other position plus the prompt.</p><p class="fine">Because the canvas is denoised as a block, 15-20 tokens leave the model per forward pass instead of one. That is the whole trick of block diffusion, and it is why this model reports 1100+ tokens per second at low batch size.</p><//>'),
 R("Then why does prefill sometimes show only half?",
   '  <details class="assumptions"><summary>Then why does the encoder pass not appear twice?</summary><p>A finished canvas is appended to the KV cache, so generation alternates: encoder pass over the new text, then a decoder pass over the next canvas. In the animation the Append phase marks that handoff; it is the same 30 blocks starting again in the other direction.</p><a href=${SOURCE_BASE} target="_blank" rel="noopener noreferrer">Read the model card</a></details>'),
 # ---------- stat grid / facts ----------
 R("<small>BACKBONE</small>",
   '  <div class="stat-grid"><div><small>PARAMETERS</small><strong>25.2<span>B total</span></strong></div><div><small>ACTIVE / TOKEN</small><strong>3.8<span>B</span></strong></div><div><small>ROUTED EXPERTS</small><strong>8<span>of 128</span></strong></div><div><small>DIFFUSION CANVAS</small><strong>256<span>positions</span></strong></div></div>'),
 R("The model card describes an optimized prefill",
   '  <${Note}>Parameters and benchmark scores come from the published model card; the 51.6476 / 27.1977 / 18.8181 GB figures come from reading the safetensors headers of the three repositories named in Sources. Nothing on this page is a shape-times-bytes estimate and no weight data is included.<//>'),
 R("Read the architecture / click to unfold",
   '  <${Section} title="Read the architecture / click to unfold" tag="30 blocks"><div class="mode-legend">${[[\'full\',\'5\',\'Global\'],[\'swa\',\'25\',\'Sliding\']].map(([key,n,label])=>html`<button class="mode-key" onClick=${()=>this.select(LAYERS.find(l=>l.mode===key).id)}><i style=${{background:COL[key]}}/><b>${n}</b><span>${label}</span></button>`)}</div>'),
 R('label="Hidden width / mHC streams"',
   '   <${Fact} label="Hidden width" value="2,816"/><${Fact} label="Attention heads / head width" value="16 / 256"/><${Fact} label="Experts per block" value="128 routed + 1 shared MLP"/><${Fact} label="Routed experts used / token" value="8 of 128"/><${Fact} label="Sliding window" value="1,024 tokens"/><${Fact} label="Global blocks" value="L05 / L11 / L17 / L23 / L29" color=${COL.full}/><p class="fine">Global means the block attends the whole context instead of a 1024-token window. All block numbers are zero-based.</p><//>'),
 R("Guided tour: Full, Reindex and Reuse",
   '  <button class="outline-btn full-width" onClick=${()=>this.patch({tab:\'guide\'})}>Guided tour: a sliding block, a global block, the canvas <${Icon} name="arrow" size=${14}/></button><${Section} title="Beyond the blocks">${[[\'vision\',\'V\',\'Vision tower + projector\',\'27 blocks / 16x16 patches / 1.1 GB\',COL.vision],[\'selfcond\',\'S\',\'Self-conditioning\',\'2816 - 2112 - 2816 / 35.7 MB\',COL.engram],[\'embed\',\'E\',\'Tied embedding\',\'262,144 x 2,816 / 1.48 GB\',COL.head]].map(([id,letter,title,sub,c])=>html`<button class="feature-link" onClick=${()=>this.select(id)}><span class="feature-icon" style=${{color:c}}>${letter}</span><span><b>${title}</b><small>${sub}</small></span><${Icon} name="chevron" size=${14}/></button>`)}<//>'),
 R("What the volumes mean",
   '  <${Section} title="What the volumes mean" tag="Storage, not VRAM"><${Fact} label="Logical values in this inventory" value=${fmtP(TOTAL_P)}/><${Fact} label=${MODE_INFO[s.precision].label+\' payload\'} value=${bytes(TOTALS[s.precision],s.binary)} color=${MODE_INFO[s.precision].color}/><p class="fine">Solids use one common scale: 1 cubic world unit = 1 GB. Empty space, paths, particles and labels are schematic. Every payload here was summed from real shard headers, not derived from shapes.</p><button class="text-link" onClick=${()=>this.setView(\'storage\')}>Inspect the byte ledger <${Icon} name="arrow" size=${15}/></button><//>'),
 # ---------- layer module panel ----------
 R("Decode step ${m.index+1} of 40.",
   ' ${m.mode&&html`<p class="layer-execution"><b>Block ${String(m.index).padStart(2,\'0\')} of 30.</b> ${m.mode===\'swa\'?\'Sliding window: 1024 tokens, 8 KV heads, then 8 of 128 experts.\':\'Global block: it attends the whole context on 2 KV heads at head_dim 512, and stores no separate v_proj.\'}</p>`}'),
 R("['Main attention queries'",
   '  [\'Queries\',\'Own weights\',\'16 query heads, 4096-wide in a sliding block and 8192-wide in a global one.\',TC.q,\'own\'],'),
 R("['Local SWA KV'",
   '  [\'Sliding KV\',m.mode===\'swa\'?\'Own weights\':\'Unused\',m.mode===\'swa\'?\'8 KV heads x 256, capped at a 1024-position window.\':\'This block attends globally instead of through the sliding window.\',TC.local,m.mode===\'swa\'?\'own\':\'off\'],'),
 R("['Global main KV'",
   '  [\'Global KV\',m.mode===\'full\'?\'Own weights\':\'Absent\',m.mode===\'full\'?\'2 global KV heads at head_dim 512: k is 1024 wide and grows with the context.\':\'No global branch in the twenty-five sliding blocks.\',TC.kv,m.mode===\'full\'?\'own\':\'off\'],'),
 R("['Indexer keys'",
   '  [\'Value projection\',m.mode===\'swa\'?\'Own weights\':\'Absent\',m.mode===\'swa\'?\'v_proj 2048 x 2816: the sliding blocks keep their own values.\':\'The five global blocks store no separate v_proj — only q, k and o are in the checkpoint.\',TC.out,m.mode===\'swa\'?\'own\':\'off\'],'),
 R("['Top-512 selection'",
   '  [\'Per-head Q/K norm\',\'Own weights\',m.mode===\'swa\'?\'256 scales for each head of q and k.\':\'512 scales for each head of q and k.\',TC.norm,\'own\']'),
 R("Borrowed objects are runtime memory",
   ' <p class="fine">Nothing here is borrowed from another block. Allocation is what differs: the sliding blocks cap their KV at 1024 positions while the five global blocks let theirs grow with the context.</p><//>`}'),
 R("title=\"Inside one expert\"",
   ' ${isLayer&&html`<${Section} title="Inside the feed-forward" tag="8 OF 128 + 1 SHARED"><div class="expert-equation">router = Wr h            \\u2192 128 logits<br/>gate, up = W1 x, W3 x   \\u2192 704 each, per expert<br/>out = \\u03a3 top-8 softmax \\u00b7 W2 (silu(gate) \\u2299 up) + shared</div><${Fact} label="One expert / logical parameters" value=${fmtP(MOE_EXPERT_P)}/><${Fact} label="Expert width" value="2,816 \\u2192 704 \\u2192 2,816"/><${Fact} label="Experts stored / block" value="128 (8 used per token)"/><${Fact} label="Fused banks / block" value=${bytes(sumB(m.ws.filter(w=>w.cat===\'expert\'),s.precision),s.binary)}/><p class="fine">The two banks are \\u00d7128 descriptors: 128 separately stored expert matrices per block, all of them on disk whether or not a token routes to them. Unfolding the bank conserves byte mass — it does not add or shrink anything.</p><button class="outline-btn full-width" onClick=${()=>this.patch({expertView:!s.expertView,expert:null,tensor:null,hoverTensor:null})}>${s.expertView?\'Return to named tensor banks\':\'Unfold all 128 individual experts\'} <${Icon} name="expand" size=${14}/></button><//>`}'),
 R("${m.id[0]==='E'&&html`<${Section} title=\"From token IDs to a gated lookup\"",
   ' ${m.id===\'selfcond\'&&html`<${Section} title="How a diffusion decoder reuses its own answer"><${Fact} label="Input" value="previous canvas estimate"/><${Fact} label="Shape" value="2,816 \\u2192 2,112 \\u2192 2,816"/><${Fact} label="Weights" value=${bytes(sumB(m.ws,s.precision),s.binary)}/><p class="fine">Each denoising step can condition on the previous step prediction. That is what self-conditioning means: the module fuses the previous estimate into the current canvas hidden state before the blocks re-predict it. The encoder pass never calls it.</p><//>`}'),
 R("Images meet the language model",
   ' ${(m.id===\'vision\'||m.id===\'aligner\')&&html`<${Section} title="Images meet the language model"><div class="expert-equation">16 \\u00d7 16 patches + a learned position table<br/>\\u2193 27 vision blocks, width 1,152, head_dim 72<br/>\\u2193 a 4,304-wide GELU-tanh MLP per block<br/>\\u2193 embed_vision [2816, 1152]<br/>2,816-channel visual embeddings</div><p class="fine">In the scene the same tensor role across the 27 blocks is combined into a bank marked \\u00d727. Those are 27 separately stored copies, not shared weights. The whole tower is 1.1 GB and stays BF16 in every published build.</p><//>`}'),
 R("NVFP4 only changes the explicitly targeted backbone routed experts",
   " <${Section} title=\"Same weights, different payload\">${Object.keys(MODE_INFO).map(mode=>{const b=sumB(m.ws,mode);return html`<div class=\"ledger-bar\"><div><span>${MODE_INFO[mode].label}</span><b>${bytes(b,s.binary)}</b></div><span><i style=${{width:b/sumB(m.ws,'bf16')*100+'%',background:MODE_INFO[mode].color,opacity:mode===s.precision?1:.35}}/></span></div>`;})}<p class=\"fine\">FP8-dynamic converts every linear weight including the experts; NVFP4 converts the fused expert banks only, so the embedding, the routers, the vision tower and the norms keep their source dtypes, and every block still stores all 128 experts.</p><//></div>`;"),
 # ---------- guide ----------
 R("Follow the data.<br/><em>Read the weights.</em>",
   ' renderGuide(){return html`<div><div class="inspect-path">A READING GUIDE TO THE ANIMATION</div><h2 class="editorial">Follow the data.<br/><em>Read the weights.</em></h2><p class="lede">The solids are learned tensors. The moving lights are explanatory signals. Do not read the size of a weight as its runtime workload.</p>'),
 R("Cross the encoder-decoder boundary",
   ' <${Section} title="Where the two passes meet"><p class="guide-copy">The encoder pass fills the KV cache; the canvas pass reads it. Because a whole canvas is denoised at once, 15-20 tokens leave the model per forward pass, and then the next encoder pass starts over the same weights. Nothing is recomputed from a second copy of the model, because there is none.</p><button class="outline-btn full-width" onClick=${()=>{this.select(\'L15\');this.patch({tab:\'inspect\'});}}>Open L15: the first block of the second role group</button><//>'),
 R("Start with a Reuse layer",
   ' <${Section} title="Start with a sliding block"><p class="guide-copy">Open L00. It is one of the twenty-five blocks with a 1024-token window: 16 query heads, 8 KV heads, per-head q and k norms, then the fused expert banks. Attention is a small fraction of it — the two expert tensors are 1.52 GB of the block at BF16.</p><button class="outline-btn full-width" onClick=${()=>{this.select(\'L0\');this.patch({tab:\'inspect\'});}}>Open L00: a sliding block</button><//>'),
 R("Compare a Full layer",
   ' <${Section} title="Compare a global block"><p class="guide-copy">Now open L05. The window is gone: 16 query heads over 2 global KV heads at head_dim 512, so q and o are 8192 wide and k is 1024. There is no separate v_proj here, and its rotary covers only a quarter of each head at theta 1e6.</p><button class="outline-btn full-width" onClick=${()=>{this.select(\'L5\');this.patch({tab:\'inspect\'});}}>Open L05: the boundary block</button><//>'),
 R("Same memory, a different search",
   ' <${Section} title="Where the 45.7 GB lives"><p class="guide-copy">Open L29 — the last global block — and switch the inspector to Experts, or press X. The bank holds 128 separate specialists: gate_up_proj [128, 1408, 2816] and down_proj [128, 2816, 704]. For any block, that is 1.52 GB of the 1.72 GB it stores at BF16.</p><button class="outline-btn full-width" onClick=${()=>{this.select(\'L29\');this.patch({expertView:true,tab:\'inspect\'});}}>Open L29: unfold all 128 experts</button><//>'),
 R("Three distinctions worth keeping",
   ' <${Section} title="Three distinctions worth keeping"><div class="layer-story"><h3>Stored is not active.</h3><p>Eight of 128 experts are chosen per token, but all 128 matrices remain in the checkpoint. Highlighting an expert changes its light, not its volume.</p><h3>Shared weights are not shared roles.</h3><p>The encoder and the decoder run the same 30 blocks with different attention masks. The only duplicated tensors are the layer scalars, one set per stack.</p><h3>A quantizer ignores more than it converts.</h3><p>The NVFP4 build converts the expert banks and leaves everything else BF16, including the 1.48 GB embedding. Read the ignore list before believing a compression ratio.</p></div><//>'),
 # ---------- storage ----------
 R("Why NVFP4 grows here",
   ' <${Section} title="Why the two quantized builds differ so much"><div class="equation"><span>EXPERTS AT BF16</span><code>2 B / parameter</code><span>FP8 / NVFP4</span><code>1.0 B / ~0.5 B + scales</code><span>WHOLE MODEL</span><code>51.65 \\u2192 27.20 \\u2192 18.82 GB</code></div><${Fact} label="FP8-dynamic saves" value=${bytes(FP8_DELTA,s.binary)} color=${COL.dec}/><${Fact} label="NVFP4 saves" value=${bytes(NV_DELTA,s.binary)} color=${COL.expert}/><p class="fine">NVFP4 is not a 4x rule: only the expert banks are converted, and the 1.48 GB tied embedding, the routers and the vision tower stay BF16. FP8-dynamic converts everything except those same protected tensors, which is why it lands at roughly half the BF16 size.</p><//>'),
 R("Accounting assumptions and limits",
   ' <details class="assumptions"><summary>Accounting assumptions and limits</summary><p>Shard headers, file padding, exporter metadata and unknown auxiliary buffers are excluded, so each figure is the tensor payload sum, not the download size.</p><p>Per-expert tensors are folded into the fused banks the BF16 checkpoint stores, and scale tensors (weight_scale, weight_scale_2, input_scale) are folded into their owner. An FP8 shard therefore shows the same logical tensor list as the BF16 one.</p><p>The embedding is a single tensor tied to the unembedding: the checkpoint has no lm_head, and nothing is double counted for it. Serving buffers, tensor parallelism and allocator overhead are different quantities entirely.</p></details></div>`;}'),
 # ---------- cache view ----------
 R("COMPRESSED SPARSE ATTENTION 2",
   ' renderCache(){const s=this.state,n=s.context,slide=KV_SLIDE_TOTAL,global=n*KV_FULL_PER_TOKEN;return html`<div><div class="inspect-path">CONTEXT COST / TWO BEHAVIOURS</div><h2 class="editorial">Twenty-five windows.<br/><em>Five readers.</em></h2><p class="lede">The sliding blocks stop at 1,024 tokens each. Only the five global blocks keep growing.</p><div class="storage-total cache-total"><small>GLOBAL CACHE / TOKEN / BF16</small><strong>10,240 <em>bytes</em></strong><span>5 blocks x 2 KV heads x 512 dims x 2 bytes — no separate v_proj</span></div>'),
 R('title="Stretch the context" tag="UP TO 1,048,576"',
   ' <${Section} title="Stretch the context" tag="UP TO 262,144"><label class="range-title" for="context-number">Context tokens <input id="context-number" type="number" min="128" max="262144" step="128" value=${n} onChange=${e=>{const v=Number(e.target.value);this.patch({context:clamp(Number.isFinite(v)?Math.round(v):128,128,262144)});}}/></label><input class="slider" aria-label="Context length logarithmic slider" type="range" min="7" max="18" step="0.05" value=${Math.log2(n)} onInput=${e=>this.patch({context:clamp(Math.round(2**Number(e.target.value)),128,262144)})}/><div class="range-ends"><span>128</span><span>8K</span><span>256K</span></div><div class="preset-row">${[16384,65536,131072,262144].map(v=>html`<button class=${n===v?\'active\':\'\'} onClick=${()=>this.patch({context:v})}>${v===262144?\'256K\':v/1024+\'K\'}</button>`)}</div><${Fact} label="Global cache payload" value=${bytes(global,s.binary)} color=${COL.full}/><${Fact} label="Sliding cache (fixed)" value=${bytes(slide,s.binary)} color=${COL.swa}/><p class="fine">The sliding figure does not depend on context: 25 blocks x 1,024 positions x 8,192 bytes. Whatever the conversation length, it stays that size.</p><//>'),
 R("Where 890 comes from",
   ' <${Section} title="Where these numbers come from"><div class="equation"><span>SLIDING / BLOCK / POSITION</span><code>(8 x 256) x k + v x 2 B = 8,192 B</code><span>GLOBAL / BLOCK / POSITION</span><code>(2 x 512) x k x 2 B = 2,048 B</code><span>FIXED SLIDING TOTAL</span><code>25 x 1,024 x 8,192 = 209.7 MB</code></div><p class="fine">Both figures are BF16 arithmetic from the published shapes: 8 KV heads at head_dim 256 with a key and a value each, against 2 global KV heads at head_dim 512 with a key only — the five global blocks store no v_proj. Quantizing the KV cache would reduce them; nothing in the checkpoint does it for you.</p><//>'),
 R("Hierarchical sparse indexer",
   ' <${Section} title="The diffusion canvas" tag="256 POSITIONS"><${Fact} label="Canvas length" value="256 positions"/><${Fact} label="Denoising steps" value="up to 48"/><${Fact} label="Tokens per forward pass" value="15-20"/><${Fact} label="Early stop" value="entropy &lt; 0.005, two agreeing steps"/><p class="fine">The canvas is not stored anywhere in the checkpoint: it is activation memory. What the blocks add is the self-conditioning module, 35.7 MB, which lets a step see the previous step estimate. The point field in the cache view illustrates 128 experts per block with 8 routed for a sample token.</p><//><${Note} color=${COL.full}>KV figures are arithmetic from the published config, not measured runtime memory, and they assume one BF16 context with no paging. Prefill workspace, allocator overhead and any quantized cache implementation are excluded.<//></div>`;}'),
 # ---------- benchmarks ----------
 R("REPORTED EVALUATIONS / NATIVE MODEL",
  ' renderBench(){const s=this.state,b=BENCH[s.bench];return html`<div><div class="inspect-path">PUBLISHER-REPORTED LEADERBOARD / AS PUBLISHED</div><h2 class="editorial">Capability,<br/><em>with context.</em></h2><p class="lede">Scores collected from each model\\u2019s own card by the original atlas page. This page measures bytes and claims no scores of its own; DiffusionGemma 26B A4B is not in that publisher set; its own card figure is added as the first row wherever the card reports the same benchmark, and a blank cell means the card does not publish it.</p><label class="field-label">Compare a benchmark<select value=${s.bench} onChange=${e=>this.patch({bench:Number(e.target.value)})}>${BENCH.map((r,i)=>html`<option value=${i}>${r[0]}</option>`)}</select></label><div class="benchmark-chart">${BENCH_MODELS.map((name,i)=>html`<div class=${\'benchmark-row\'}><div><span>${name}</span><b>${b[2][i]===null?\'Not reported\':b[2][i].toFixed(1)}</b></div><div class="bench-track"><i style=${{width:(b[2][i]||0)+\'%\',background:COL.enc,opacity:1}}/></div></div>`)}</div><p class="fine">${b[0]} / ${b[1]}. Percentages or the card\'s 0-100 score; ZeroBench uses Pass@5, ProgramBench Almost@1, DeepSWE resolved rate. Do not average unlike metrics.</p>'),
 R("Evaluation conditions",
   ' <${Section} title="Evaluation conditions"><${Fact} label="Sampling" value="T 0.8 \\u2192 0.4"/><${Fact} label="Entropy bound" value="0.1"/><${Fact} label="Max denoising steps" value="48"/><${Fact} label="Context" value="up to 256K tokens"/><p class="fine">Harnesses and protocols differ per benchmark; MMMU Pro and MATH-Vision exercise the vision tower through the same 30 blocks. See the model card for task-specific settings.</p><${Note} color=${COL.full}>Publisher-reported scores. No quality equivalence is claimed for the FP8 or NVFP4 builds — they are byte-accounting subjects here, not evaluated variants.<//><//>'),
 R("Reasoning is an API setting",
   ' <${Section} title="The sampler is the API setting"><label class="range-title">temperature at step ${s.effort<50?\'1 of 48\':\'48 of 48\'} <output>${(0.8-0.4*s.effort/100).toFixed(2)}</output></label><input class="slider" type="range" min="1" max="100" value=${s.effort} aria-label="Denoising progress example" onInput=${e=>this.patch({effort:Number(e.target.value)})}/><pre class="small-code">${JSON.stringify({temperature:Number((0.8-0.4*s.effort/100).toFixed(2)),entropy_bound:0.1,max_steps:48,canvas:256},null,2)}</pre><p class="fine">Temperature falls linearly from 0.8 to 0.4 across the denoising steps, and positions whose mutual-information bound exceeds 0.1 are renoised rather than accepted. This slider only illustrates that schedule.</p><//></div>`;}'),
 # ---------- sources ----------
 R("PROVENANCE / SEPTEMBER 10, 2026",
   ' renderSources(){return html`<div><div class="inspect-path">PROVENANCE / MEASURED FROM PUBLISHED SHARDS</div><h2 class="editorial">Trace every<br/><em>assumption.</em></h2><p class="lede">The card supplies the claims. The config supplies the shapes. The shard headers supply the bytes — all three modes on this page were read over HTTP Range, not estimated.</p><div class="source-status"><i/> Offline app / no runtime requests</div>'),
 R("The supplied NVFP4 recipe",
   ' <${Section} title="How the two quantized builds differ"><p class="fine">RedHatAI ships compressed-tensors W8A8 in one shard: e4m3 linear weights with per-channel scales, per-token dynamic activation scales, and a 255-entry ignore list that keeps routers, layer_scalar tensors and RMSNorms in BF16. nvidia ships modelopt NVFP4 in two shards whose ignore list narrows to <code>lm_head, embed_vision, *mlp*, *router*, *self_attn*, *self_conditioning*, *vision_tower</code> — the fused expert banks are the only converted tensors.</p><${Fact} label="FP8-dynamic" value="27.1977 GB / 1 shard"/><${Fact} label="NVFP4" value="18.8181 GB / 2 shards"/><${Fact} label="Embedding in both" value="BF16 / 1.48 GB"/><${Fact} label="NVFP4 scale groups" value="16 weights + F32 per expert"/><//>'),
 R("Tower, split and execution",
   ' <${Section} title="Tower, split and the two passes"><p class="fine">The tower is a logical block-order view, not a map of safetensors byte offsets. Splitting moves the same 30 objects into two groups; it does not create a second model, and both layouts use the exact same tensor inventory.</p><p class="fine">The default animation starts in Denoise and loops that phase. Encode, Canvas, Denoise, Append and Repeat are the five stages this page uses to describe block diffusion; the sampler numbers (256 positions, up to 48 steps, entropy bound 0.1, early stop at 0.005) are taken from the model card. Nothing here executes a model.</p><p class="fine">Training mode is a generic forward/loss/backward schematic. It does not reproduce the training recipe of this model and it never models optimizer memory as checkpoint bytes.</p><//>'),
 R("Interpretation rules",
   ' <${Section} title="Interpretation rules"><p class="fine">25.2B total parameters and 3.8B active per token are the published claims. The modeled total here is ${fmtP(TOTAL_P)} stored logical values across 1,047 tensors, audited tensor by tensor from the shard headers of the three repositories named above.</p><p class="fine">Routing, flow, canvas and step animations are explanatory. No inference runs. The timeline is not a latency chart and no weight data is bundled in this HTML.</p><p class="fine">Volumes, not footprint areas: unfolding experts conserves modeled mass, and zooming never alters the byte scale. KV cache sizes shown in the context view are arithmetic from published shapes, not measured device memory.</p><//>'),
 # ---------- modals / chrome ----------
 R("DEEPSEEK MODEL ATLAS",
   ' renderModal(){const s=this.state;if(!s.modal)return null;let title={help:\'Explore the model\',config:\'Embedded configurations\',audit:\'Safetensors header audit\',credits:\'Credits & licenses\'}[s.modal];return html`<div class="modal-backdrop" onClick=${e=>{if(e.target===e.currentTarget)this.patch({modal:null});}}><div class=${\'modal \'+(s.modal===\'config\'?\'wide\':\'\')} role="dialog" aria-modal="true" aria-label=${title}><div class="modal-head"><div><div class="eyebrow">DIFFUSIONGEMMA MODEL ATLAS</div><h2>${title}</h2></div><button class="icon-button" aria-label="Close dialog" onClick=${()=>this.patch({modal:null})}><${Icon} name="close"/></button></div><div class="modal-body">'),
 R("['P','Cycle native / NVFP4 / BF16']",
   ' ${s.modal===\'help\'&&html`<p class="lede">A file to keep, inspect and share. Everything needed to render the atlas is embedded.</p><div class="help-grid">${[[\'V\',\'Merge tower / separate the two role groups\'],[\'B / T\',\'Hide inspector / animation panel\'],[\'+ / -\',\'Increase / decrease all text\'],[\'Blank panel background\',\'Click the arrow cursor to collapse; text is protected\'],[\'Drag\',\'Orbit the model\'],[\'Right-drag / Shift-drag\',\'Pan in camera space\'],[\'Wheel / pinch\',\'Zoom without changing scale\'],[\'Click a block\',\'Open named weight tensors\'],[\'Click a labeled weight\',\'Pin its role, shape and bytes\'],[\'F / X\',\'Columns or floor / banks or experts\'],[\'[ / ]\',"Walk the block\\u2019s tensor list"],[\'Click an expert\',\'Split its gate-up and down bank\'],[\'Two-finger drag\',\'Pan on touch screens\'],[\'R / double-click\',\'Reset the camera\'],[\'P\',\'Cycle BF16 / FP8 / NVFP4\'],[\'Space\',\'Pause or resume the story\'],[\'Up / Down\',\'Previous or next module\'],[\'L / K\',\'Toggle labels / flow paths\'],[\'1 / 2 / 3\',\'Architecture / Storage / Context\']].map(([k,v])=>html`<div><kbd>${k}</kbd><span>${v}</span></div>`)}</div><${Note}>The animation starts in Denoise and loops that phase. Choose another phase or turn off Loop phase to play the whole story. Tower order is not disk order; Split moves the same block objects. Training mode illustrates gradients, not a second checkpoint. Routing is synthetic. Solids represent audited stored bytes, not device utilization, FLOPs or measured throughput.<//><button class="outline-btn full-width" onClick=${()=>this.patch({modal:\'audit\'})}>Check exact bytes from local checkpoint headers</button>`}'),
 R("Full supplied fields.",
   ' ${s.modal===\'config\'&&html`<div class="config-switch">${[\'bf16\',\'fp8\',\'nvfp4\'].map(m=>html`<button class=${(s.configMode||\'bf16\')===m?\'active\':\'\'} onClick=${()=>this.patch({configMode:m})}>${m===\'bf16\'?\'Released config\':m===\'fp8\'?\'FP8-dynamic build\':\'NVFP4 build\'}</button>`)}</div><p class="fine">The released config.json is embedded field for field. The two quantized tabs show the published quantization_config of those repositories, never a rewritten recipe.</p><pre class="config-code">${JSON.stringify(s.configMode===\'nvfp4\'?NV_CONFIG:s.configMode===\'fp8\'?FP8_CONFIG:CFG,null,2)}</pre><button class="outline-btn" onClick=${()=>safeDownload((s.configMode||\'bf16\')+\'-config.json\',JSON.stringify(s.configMode===\'nvfp4\'?NV_CONFIG:s.configMode===\'fp8\'?FP8_CONFIG:CFG,null,2))}><${Icon} name="download" size=${16}/> Save config.json</button>`}'),
 # ---------- scene chrome ----------
 R("DSpark draft stages",
   ' <div class="workspace"><nav class="layer-rail" aria-label="Model modules"><button class=${\'rail-special\'+(s.selected===\'selfcond\'?\' selected\':\'\')} title="Self-conditioning module" onClick=${()=>{this.patch({view:\'architecture\'});this.select(\'selfcond\');}}>S</button><div class="rail-caption">OUT</div><div class="rail-layers">${[...LAYERS].reverse().map(m=>html`<button key=${m.id} aria-label=${m.label+\' \'+m.mode} class=${\'rail-layer \'+(s.selected===m.id?\'selected\':\'\')} style=${{ \'--layer-color\':COL[m.mode]}} onClick=${()=>{if(s.view!==\'architecture\')this.patch({view:\'architecture\'});this.select(m.id);}} title=${m.label+\' / \'+m.mode+\' / \'+m.part}><span>${String(m.index).padStart(2,\'0\')}</span><i/></button>`)}</div><div class="rail-caption">IN</div><button class=${\'rail-special\'+(s.selected===\'vision\'?\' selected\':\'\')} title="Vision tower" onClick=${()=>{this.patch({view:\'architecture\'});this.select(\'vision\');}}>V</button></nav>'),
 R("global KV payload",
   ' <div class="scene-scale"><span class="scale-glyph"/><span>1 cubic unit = 1 GB <em>${s.view===\'cache\'?\'context cache arithmetic\':\'audited weight payload\'} / ${this.engine&&this.engine.renderer.software?\'CPU 3D\':\'WebGL2\'}</em></span></div><div class="scene-legend">${(s.view===\'cache\'?[[\'full\',\'Global blocks\'],[\'swa\',\'Sliding\']]:s.view===\'storage\'?[[\'expert\',\'Experts\'],[\'vision\',\'Vision\'],[\'attn\',\'Attention\']]:[[\'full\',\'Global\'],[\'swa\',\'Sliding\']]).map(([key,l])=>html`<span><i style=${{background:COL[key]}}/>${l}</span>`)}</div>'),
 R("DSPARK / FIVE-TOKEN DRAFT",
   ' ${s.view===\'architecture\'&&s.activity===\'inference\'&&(phaseI===2||s.selected===\'selfcond\')&&html`<div class="draft-diagram"><div><b>CANVAS / 256 POSITIONS</b><small>Schematic confidence across denoising steps</small></div><div class="draft-cells">${[.98,.94,.87,.62,.41].map((confidence,i)=>{let step=(s.time>=11?s.time-11:2),stage=step<1.2?\'masked\':step<2.8?\'scored\':i<3?\'kept\':i===3?\'renoise\':\'redraft\';return html`<div class=${"draft-cell "+stage}><small>pos ${i*51+7}</small><b>${step<1.2?\'...\':confidence.toFixed(2)}</b><span>${stage}</span></div>`;})}</div><p>Canvas positions, not weight volumes. No measured acceptance rate.</p></div>`}'),
 R("function fontPreference(){",
   'function fontPreference(){try{const n=Number(localStorage.getItem(\'diffusiongemma-atlas-font\'));return Number.isFinite(n)&&n>=.85&&n<=1.5?n:1;}catch(_){return 1;}}'),
 R("componentDidUpdate(prevProps,prevState)",
   ' componentDidUpdate(prevProps,prevState){if(prevState&&prevState.fontScale!==this.state.fontScale){try{localStorage.setItem(\'diffusiongemma-atlas-font\',String(this.state.fontScale));}catch(_){}}if(this.engine)this.engine.sync(this.state);if(prevState&&(prevState.selected!==this.state.selected||prevState.view!==this.state.view||prevState.tab!==this.state.tab)&&this.panelRef.current)this.panelRef.current.scrollTop=0;if(prevState&&prevState.modal!==this.state.modal){if(this.state.modal){this.modalReturnFocus=document.activeElement;setTimeout(()=>document.querySelector(\'.modal-head button\')?.focus(),0);}else if(this.modalReturnFocus&&document.contains(this.modalReturnFocus))this.modalReturnFocus.focus({preventScroll:true});}}'),
 R("deepseek-v41-'+mode+'-derived-ledger.json'",
   ' exportLedger(){let mode=this.state.precision;safeDownload(\'diffusiongemma-26b-\'+mode+\'-ledger.json\',JSON.stringify({title:\'DiffusionGemma 26B A4B - measured safetensors payload ledger\',date:\'2026-09-14\',precision:mode,status:\'MEASURED FROM PUBLISHED SHARD HEADERS\',notes:[\'Tensor payload only. File padding, headers, exporter metadata and unknown buffers are excluded.\',\'Per-expert tensors are folded into the fused banks of the BF16 checkpoint; scale tensors are folded into their owner.\',\'KV cache figures shown in the context view are arithmetic from published shapes, not device memory.\'],bytes:TOTALS[mode],logicalParameters:TOTAL_P,modules:MODULES.map(m=>({id:m.id,label:m.label,bytes:sumB(m.ws,mode),parameters:sumP(m.ws),weights:m.ws.map(w=>({...w,storageBytes:wBytes(w,mode),displayFormat:wFormat(w,mode)}))})),sources:SOURCES},null,2));this.notify(\'Byte ledger exported. Every figure is measured from published shard headers.\');}'),
 R("deepseek-v41-3d-",
   ' snapshot(){if(!this.engine)return;const a=document.createElement(\'a\');a.href=this.engine.snapshot();a.download=\'diffusiongemma-26b-3d-\'+this.state.view+\'.png\';a.click();this.notify(\'3D canvas saved. Interface labels are not part of the canvas.\');}'),
 R("['native','nvfp4','bf16']",
   ' key(e){if(e.ctrlKey||e.metaKey||e.altKey)return;const k=e.key.toLowerCase();if(k===\'escape\'){if(this.state.modal||this.state.sheet)this.patch({modal:null,sheet:false});else if(this.state.tensor||this.state.expert!==null)this.patch({tensor:null,hoverTensor:null,expert:null});else this.patch({selected:null});return;}if(this.state.modal){if(k===\'tab\'){let nodes=[...document.querySelectorAll(\'.modal button:not(:disabled),.modal a,.modal input:not([hidden]),.modal select,.modal summary\')].filter(x=>x.getClientRects().length),first=nodes[0],last=nodes[nodes.length-1];if(e.shiftKey&&document.activeElement===first){e.preventDefault();last?.focus();}else if(!e.shiftKey&&document.activeElement===last){e.preventDefault();first?.focus();}}return;}if(/input|select|textarea/i.test(e.target.tagName)||e.target.tagName===\'BUTTON\'&&k===\' \')return;if(k===\' \'){e.preventDefault();this.patch({playing:!this.state.playing});}if(k===\'arrowdown\'){e.preventDefault();this.walk(1);}if(k===\'arrowup\'){e.preventDefault();this.walk(-1);}if(k===\'+\'||k===\'=\'){e.preventDefault();this.changeFont(.1);}if(k===\'-\'){e.preventDefault();this.changeFont(-.1);}if(k===\'b\')this.togglePanel();if(k===\'t\')this.patch({timelineHidden:!this.state.timelineHidden});if(k===\'v\')this.setLayout(this.state.layout===\'tower\'?\'split\':\'tower\');if(k===\'r\'&&this.engine){if(this.state.selected)this.engine.focus(this.state.selected);else this.engine.reset();}if(k===\'f\')this.patch({detailLayout:this.state.detailLayout===\'floor\'?\'column\':\'floor\'});if(k===\'x\')this.patch({expertView:!this.state.expertView,expert:null});if(k===\'[\')this.walkTensor(-1);if(k===\']\')this.walkTensor(1);if(k===\'p\'){let keys=MODE_KEYS;this.patch({precision:keys[(keys.indexOf(this.state.precision)+1)%keys.length]});}if(k===\'l\')this.patch({labels:!this.state.labels});if(k===\'k\')this.patch({links:!this.state.links});if([\'1\',\'2\',\'3\'].includes(k))this.setView([\'architecture\',\'storage\',\'cache\'][Number(k)-1]);}'),
 # ---------- document chrome ----------
 R('<meta name="description" content="Offline Preact and WebGL2 architecture atlas for DeepSeek',
   '<meta name="description" content="Offline Preact and WebGL2 architecture atlas for DiffusionGemma 26B A4B: a block-diffusion Gemma 4 MoE with 30 blocks, 128 experts per block, a 27-block vision tower, a diffusion canvas and three shard-audited precision modes (BF16, FP8, NVFP4).">'),
 R("<title>DeepSeek V4.1 Flash - Tensor Atlas v4</title>",
   ' <title>DiffusionGemma 26B A4B - Tensor Atlas v4</title>'),
 R("DeepSeek Model Atlas requires JavaScript",
   '<div id="root"></div><noscript><div style="padding:40px;color:#d7e6ff;font-family:monospace">DiffusionGemma Model Atlas requires JavaScript. Everything needed is already embedded in this file. Enable JavaScript and reload.</div></noscript>'),
 R("/* DeepSeek V4.1 Flash Atlas. Model facts from the supplied configs and",
   '/* DiffusionGemma 26B A4B Atlas. Model facts from the published config.json; every byte was'),
 R("DeepSeek's reference source; byte totals are schema-derived, not shard-audited. */",
   "   measured from the safetensors headers of the BF16, FP8-dynamic and NVFP4 repositories. */"),
 R("function moduleExplanation(m){",
   ' function moduleExplanation(m){if(!m)return EXP.overview;if(m.mode)return EXP[m.mode];if(m.id===\'vision\'||m.id===\'aligner\')return EXP.vision;if(m.id===\'selfcond\')return EXP.selfcond;if(m.id===\'embed\')return {title:\'Token embedding, tied to the unembedding.\',body:\'[262,144, 2,816] BF16: 1.48 GB, the entire vocabulary. Because tie_word_embeddings is true there is no lm_head tensor in the checkpoint at all, and both quantized builds keep this table in BF16.\'};if(m.id===\'head\')return {title:\'Final RMSNorm.\',body:\'One 2,816-wide RMSNorm on the residual stream before the tied unembedding. A norm, not a vocabulary matrix: the head itself is tied and already counted in the embedding.\'};return EXP.overview;}'),
 R("function describeFormat(w){",
   ' function describeFormat(w){return w.note||({bf16:\'Two bytes per logical value in the released BF16 checkpoint.\',fp8:\'e4m3 weights with one per-channel scale (plus per-token dynamic activation scales) in the RedHatAI compressed-tensors build.\',nvfp4:\'Block-16 NVFP4: e2m1 values with one e4m3 scale per 16 weights, plus a per-tensor F32 scale. The nvidia build applies it to the fused expert banks only.\'}[w.format]||\'See the embedded configurations in Sources.\');}'),
 R("function resolveAuditModule(name){",
   ' function resolveAuditModule(name){let n=name.replace(/^(?:model\\.)+/,\'\');let a=n.match(/^decoder\\.layers\\.(\\d+)\\./);if(a)return \'L\'+a[1];if(/^encoder\\.vision_tower/.test(n))return \'vision\';if(/^encoder\\.embed_vision/.test(n))return \'aligner\';if(/^encoder\\.language_model/.test(n))return \'L* (encoder scales)\';if(/^decoder\\.self_conditioning/.test(n))return \'selfcond\';if(/^decoder\\.embed_tokens/.test(n))return \'embed\';if(/^decoder\\.norm|^decoder\\.lm_head|^lm_head/.test(n))return \'head\';return \'Unmapped\';}')
]


LITERAL_SUBS: list[tuple[str, str]] = [
    ("SCHEMA-DERIVED PAYLOAD",
     "MEASURED FROM SHARD HEADERS"),
    ("<p class=\"guide-copy\">The ledger remains schema-derived. A complete set of checkpoint headers is required before calling its totals exact on-disk measurements.</p>",
     "<p class=\"guide-copy\">Every total on this page is measured: the three checkpoints were read header by header over HTTP Range, tensor by tensor, and the storage view sums exactly those bytes. The audit panel in Sources lets you run the same arithmetic on shards you already have on disk.</p>"),
    ("<${Note} color=${s.precision==='nvfp4'?COL.dec:COL.enc}><strong>${s.precision==='bf16'?'A reference, not a checkpoint.':EXP[s.precision].title}</strong> ${s.precision==='bf16'?'Every logical value is charged two bytes. This baseline is hypothetical, not an available BF16 repository or a device-memory prediction.':EXP[s.precision].body}<//>",
     "<${Note} color=${MODE_INFO[s.precision].color}><strong>${EXP[s.precision].title}</strong> ${EXP[s.precision].body}<//>"),
    ("<span>Schema-derived tensor payload estimate</span>",
     "<span>Measured: every shard header read, tensor by tensor</span>"),
    ("aria-label=\"DeepSeek V4.1 Flash architecture explorer home\"",
     "aria-label=\"DiffusionGemma 26B A4B architecture explorer home\""),
    ("<h1>DeepSeek <em>V4.1 Flash</em>",
     "<h1>DiffusionGemma <em>26B A4B</em>"),
    ("let ids=ph.name==='Encode'?Array.from({length:20},(_,i)=>'L'+i):ph.name==='Canvas'?Array.from({length:20},(_,i)=>'L'+(20+i)):['embed',...LAYERS.map(m=>m.id),'head'];",
     "let ids=ph.name==='Encode'?Array.from({length:15},(_,i)=>'L'+i):['embed',...LAYERS.map(m=>m.id),'head'];"),
    ("for(let i=1;i<ids.length;i++)this.line(V3.add(pos(ids[i-1]),offset),V3.add(pos(ids[i]),offset),ph.color,.16);",
     "for(let i=1;i<ids.length;i++){const pa=pos(ids[i-1]),pb=pos(ids[i]);if(!pa||!pb)continue;this.line(V3.add(pa,offset),V3.add(pb,offset),ph.color,.16);}"),
    ("const frac=train?clamp(progress+lane*.009,0,.999):((progress*(ph.name==='Denoise'?2:1)+lane*.045)%1),at=frac*(ids.length-1),lo=Math.floor(at),hi=Math.min(lo+1,ids.length-1),a=pos(ids[lo]),b=pos(ids[hi]);this.point(a.map((v,k)=>lerp(v,b[k],at-lo)+offset[k]),ph.color,9,.9);",
     "const frac=train?clamp(progress+lane*.009,0,.999):((progress*(ph.name==='Denoise'?2:1)+lane*.045)%1),at=frac*(ids.length-1),lo=Math.floor(at),hi=Math.min(lo+1,ids.length-1),a=pos(ids[lo]),b=pos(ids[hi]);if(a&&b)this.point(a.map((v,k)=>lerp(v,b[k],at-lo)+offset[k]),ph.color,9,.9);"),
    ("const index=m.mode?m.index%15:0,delay=index/19*.10",
     "const index=m.mode?m.index%15:0,delay=index/14*.10"),
    ("stackKey(m){if(false)return null;",
     "stackKey(m){let mk=null;if(m.id==='embed')mk={chain:0,index:-1};else if(m.mode)mk={chain:m.index<15?0:1,index:m.index};else if(m.id==='head')mk={chain:1,index:30};return mk;"),
    ("if(p.name==='Forward')return Math.exp(-Math.pow((i/40-t)*7,2))*.8;",
     "if(p.name==='Forward')return Math.exp(-Math.pow((i/30-t)*7,2))*.8;"),
    ("if(this.state.view==='storage')this.storage();else if(this.state.view==='cache')this.cache();else this.architecture();",
     "try{if(this.state.view==='storage')this.storage();else if(this.state.view==='cache')this.cache();else this.architecture();}catch(err){if(!this.viewError){this.viewError=1;console.error('atlas: '+this.state.view+' view failed:',err);}}"),
    ("<div class=\"format-notice\"><b>+${bytes(NV_DELTA,s.binary)}</b> modeled vs BF16 / smaller scale groups</div>",
     "<div class=\"format-notice\"><b>${bytes(TOTALS.bf16-TOTALS[s.precision],s.binary)} smaller</b> than the BF16 checkpoint / measured from shard headers</div>"),
]


def patch_code(text: str) -> tuple[str, list[str]]:
    """Targeted replacements inside the renderer's engine code (not whole lines)."""
    report: list[str] = []
    subs = [
        # mode key: the engine called the reference checkpoint "native"
        (r"'native'", "'bf16'", 0),
        (r"precision:'bf16'", "precision:'bf16'", 0),
        # deck split: first fifteen blocks are the encoder-side group
        (r"m\.index<20\?", "m.index<15?", 0),
        (r"m\.index%20", "m.index%15", 0),
        (r"CT\.compress_ratios\[i\]", "1", 0),
        (r"b/sumB\(m\.ws,'bf16'\)", "b/sumB(m.ws,'bf16')", 0),
        # scene anchors for 30 blocks
        (r"pos\('L19'\)", "pos('L14')", 0),
        (r"pos\('L36'\)", "pos('L29')", 0),
        (r"pos\('L20'\)", "pos('L15')", 0),
        (r"pos\('D2'\)", "pos('L29')", 0),
        (r"'L19 -> L20'", "'L14 -> L15'", 0),
        (r"'ENCODER / L00-L19'", "'ENCODER ROLE / L00-L14'", 0),
        (r"'DECODER / L20-L39'", "'DECODER ROLE / L15-L29'", 0),
        (r"'ONE CHECKPOINT / L00-L39'", "'ONE WEIGHT SET / L00-L29'", 0),
        (r"'First 20 layers / runs in decode'", "'same weights, causal pass'", 0),
        (r"'Next 20 layers / runs in decode'", "'same weights, bidirectional canvas'", 0),
        (r"'Ordered layers, not physical shard offsets\.'", "'30 blocks, first encoder-side, then decoder-side.'", 0),
        (r"'Optimized prefill: prepare decoder KV'", "'Encoder pass: build the KV cache'", 0),
        (r"'Forward pass continues\. No model swap\.'", "'The next pass continues over the same blocks.'", 0),
        (r"for\(const m of LAYERS\)\{const p=pos\(m\.id\);this\.label\(V3\.add\(p,\[2\.5,0,1\.9\]\),'L'\+String\(m\.index\)\.padStart\(2,'0'\),'',m\.index<20\?COL\.enc:COL\.dec,1\);\}",
         "for(const m of LAYERS){const p=pos(m.id);this.label(V3.add(p,[2.5,0,1.9]),'L'+String(m.index).padStart(2,'0'),'',m.index<15?COL.enc:COL.dec,1);}", 0),
        # flow animation: the phases this atlas actually has
        (r"'SWA replay'", "'Canvas'", 0),
        (r"'Prefill'", "'Encode'", 0),
        (r"'Decode'", "'Denoise'", 0),
        (r"if\(train\)\{const hp=pos\('head'\);if\(ph\.name==='Loss'\).*?COL\.full,10\);\}",
         "if(train){const hp=pos('head');if(ph.name==='Loss')this.label(V3.add(hp,[4,.6,0]),'PREDICTION vs TARGET','Canvas scored at once (schematic)',COL.engram,10);if(ph.name==='Backward')this.label(V3.add(pos('L15'),[4,0,0]),'GRADIENTS','Backward path, not reverse inference.',COL.mhc,10);}", 0),
        (r"if\(!train&&ph\.name==='DSpark'\)for\(let i=0;i<3;i\+\+\)this\.curve\(pos\('L'\+\(37\+i\)\),pos\('D'\+i\),TC\.mhc,\.55,i\*\.3,3,2\);",
         "if(!train&&ph.name==='Denoise'){const amp=V3.add(pos('L29'),[0,1.6,0]);this.label(amp,'CANVAS 256','positions denoised in parallel',COL.dec,9);this.curve(amp,pos('selfcond'),TC.mhc,.5,.3,0.6,2);}", 0),
        (r"if\(!train&&ph\.name==='Encode'\)", "if(!train&&ph.name==='Encode')", 0),
        # dependency pairs in the expanded block
        (r"const pairs=\[\[.*?\]\];",
         "const pairs=[['self_attn.q_proj.weight','self_attn.q_norm.weight'],['self_attn.k_proj.weight','self_attn.k_norm.weight'],['router.proj.weight','experts.gate_up_proj'],['experts.gate_up_proj','experts.down_proj'],['mlp.gate_proj.weight','mlp.down_proj.weight']];", 0),
        (r"w\.name\.includes\('\.w1\.'\)\?'W1 / gate':w\.name\.includes\('\.w3\.'\)\?'W3 / up':'W2 / down'",
         "w.name.includes('gate_up')?'gate + up bank':'down bank'", 0),
        # expert grid: 128 experts, 8 active, no 384/128 branching
        (r"pickExperts\(m\.index\*1000\+Math\.floor\(this\.clock\*\.65\),g\.n,g\.n===128\?3:6\)",
         "pickExperts(m.index*1000+Math.floor(this.clock*.65),g.n,8)", 0),
        (r"'Expert '\+String\(i\)\.padStart\(3,'0'\)", "'Expert '+String(i).padStart(3,'0')+' of 128'", 0),
        # stack/deck geometry for 30 blocks
        (r"index:m\.index\+1\}", "index:m.index+1}", 0),
        (r"if\(m\.id==='head'\)return \{chain:1,index:40\};", "if(m.id==='head')return {chain:1,index:30};", 0),
        (r"chain:m\.index<20\?0:1", "chain:m.index<15?0:1", 0),
        # storage / cache views
        (r"this\.emit\('category:'\+id,p,b,col,\{module:'cat:'\+id,label:name,bytes:b\},\{style:id==='engram'\?2:0,glow:id==='routed'\?\.06:0\}\)",
         "this.emit('category:'+id,p,b,col,{module:'cat:'+id,label:name,bytes:b},{style:(id==='vision'||id==='selfcond')?2:0,glow:id==='expert'?.06:0})", 0),
        (r"cache\(\)\{const s=this\.state,ids=\[2,8,14,20\];", "cache(){const s=this.state,ids=CT.kv_source_layer_ids;", 0),
        (r"let n=id<20\?Math\.floor\(s\.context/2\):s\.context,b=n\*356",
         "let n=s.context,b=n*2*2*512*2", 0),
        (r"i===3\?COL\.dec:COL\.full", "COL.full", 0),
        (r"\('L'\+String\(id\)\.padStart\(2,'0'\),\('2:1 / ':'1:1 / '\)\+bytes\(b,s\.binary\)",
         "this.label([p[0],2.0+side/2,0],'L'+String(id).padStart(2,'0'),'global KV / '+bytes(b,s.binary)", 0),
        (r"const selected=pickExperts\(Math\.floor\(this\.clock\*\.4\),2048,512\),set=new Set\(selected\);",
         "const selected=pickExperts(Math.floor(this.clock*.4),128,8),set=new Set(selected);", 0),
        (r"for\(let i=0;i<2048;i\+\+\)\{let x=\(i%64-31\.5\)\*\.077,y=-\.5-Math\.floor\(i/64\)\*\.052;this\.point\(\[x,y,0\],set\.has\(i\)\?COL\.dec:'#4a6280',set\.has\(i\)\?3\.3:2,set\.has\(i\)\?\.72:\.22\);\}",
         "for(let i=0;i<128;i++){let x=(i%32-15.5)*.155,y=-.5-Math.floor(i/32)*.104;this.point([x,y,0],set.has(i)?COL.dec:'#4a6280',set.has(i)?3.3:2,set.has(i)?.72:.22);}", 0),
        (r"'2,048 CANDIDATE BLOCKS x 8'", "'128 EXPERTS PER BLOCK'", 0),
        (r"'A schematic pool; the true cap is 16,384 candidate positions\.'", "'Eight of them fire for a sample token; all 128 stay on disk.'", 0),
        (r"'TOP-512 POSITIONS PER QUERY'", "'8 ROUTED OF 128'", 0),
        (r"'512 illuminated sample marks illustrate sparse selection\.'", "'Illuminated marks show one token\\u2019s routing, not extra bytes.'", 0),
        (r"'FOUR SHARED GLOBAL BANKS'", "'FIVE GLOBAL BLOCKS'", 0),
        (r"'890 B / original token across the model, not per layer\.'", "'10,240 B per token in BF16, and they grow with the context.'", 0),
        (r"'THE CHECKPOINT, BY VOLUME','One cubic world unit = one decimal GB\. No category is enlarged\.'",
         "'THE CHECKPOINT, BY VOLUME','One cubic world unit = one decimal GB. Measured payloads, not estimates.'", 0),
        # audit panel wording
        (r"or silently replace the estimated 3D scene", "or silently replace the measured 3D scene", 0),
        (r"Model data is attributed to DeepSeek-AI's card, configuration and reference implementation\. The model card identifies the repository and weights as MIT-licensed\.",
         "Model data is attributed to Google's DiffusionGemma model card and published configuration; the three byte inventories come from the safetensors headers of google/diffusiongemma-26B-A4B-it, RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic and nvidia/diffusiongemma-26B-A4B-it-NVFP4. The weights are Apache-2.0 (Gemma 4 licence); no weight data is bundled here.", 0),
        # palette keys that still carry the original model's vocabulary
        (r"COL\.engram", "COL.auxa", 0),
        (r"TC\.engram", "TC.auxa", 0),
        (r"COL\.mhc", "COL.auxb", 0),
        (r"TC\.mhc", "TC.auxb", 0),
        (r"engram:'#", "auxa:'#", 0),
        (r"mhc:'#", "auxb:'#", 0),
        # module colour chain: no E/D modules in this model
        (r"m\.id\[0\]==='E'\?TC\.auxa:m\.id\[0\]==='D'\?TC\.auxb", "m.id==='selfcond'?TC.selfcond", 0),
        (r"m\.id\[0\]==='E'\?\{glow:\.02,style:2,dim\}", "m.id==='selfcond'?{glow:.02,style:2,dim}", 0),
        (r"glow:m\.index<40\?this\.layerHeat\(m\.index\):ph==='Denoise'\?\.42:0", "glow:this.layerHeat(m.index)", 0),
        (r"glow:m\.index<40\?this\.layerHeat\(m\.index\):ph==='DSpark'\?\.42:0", "glow:this.layerHeat(m.index)", 0),
        (r"'Decode / all 40 layers'", "'Denoise / 30 blocks'", 0),
        (r"'Optimized prompt prefill'", "'Encoder pass over the prompt'", 0),
        (r"'deepseek-exact-selected-shard-audit\.json'", "'diffusiongemma-26b-shard-audit.json'", 0),
        (r"if\(m\.id\.startsWith\('E'\)&&ok\)", "if(false&&ok)", 0),
        (r"else if\(m\.id\[0\]==='E'\)\{tower=", "else if(false){tower=", 0),
        (r"if\(m\.id\[0\]==='D'\)\{tower=\[0,29\.1\+\(m\.index-40\)\*\.73,0\];split=\[5\.4,17\.6\+\(m\.index-40\)\*\.73,0\];\}",
         "if(false){tower=[0,0,0];split=[0,0,0];}", 0),
        (r"stackKey\(m\)\{", "stackKey(m){if(false)return null;", 0),
        (r"const n=m\.id\.startsWith\('D'\)\?128:384", "const n=CT.num_experts", 0),
        (r"\$\{selected\.id\.startsWith\('D'\)\?'128':'384'\} experts", "128 experts", 0),
        (r"n\+' EXPERTS / TOP-'\+\(n===128\?3:6\)", "n+' EXPERTS / TOP-'+CT.top_k_experts", 0),
        (r"\bENGRAM\b", "AUX_TABLES", 0),
        (r"L00-L39", "L00-L29", 0),
        (r"'One forward pass, shown in two groups'", "'One set of blocks, two passes'", 0),
        (r"'The checkpoint in layer order'", "'The checkpoint in block order'", 0),
        (r"modeled vs native / smaller scale groups", "modeled vs BF16 / smaller scale groups", 0),
        (r"aria-label=\"Backbone layout\"", "aria-label=\"Block layout\"", 0),
        (r"20 \+ 20", "15 + 15", 0),
        (r"'Decode: L00-L19 then L20-L39 then head\. No layers skipped\.'", "'Encoder pass, then the canvas, then the same blocks again.'", 0),
        (r"'L00-L39 / one inventory / click any layer to unfold\.'", "'L00-L29 / one inventory / click any block to unfold.'", 0),
        (r"'Decode / 15 blocks'", "'Denoise / 30 blocks'", 0),
    ]
    for pat, rep, expect in subs:
        text, n = re.subn(pat, lambda _m, rep=rep: rep, text)
        if expect and n != expect:
            raise SystemExit(f"code patch {pat!r}: {n} substitutions, expected {expect}")
        report.append(f"{n:>3}x  {pat[:70]}")
    for old, new in LITERAL_SUBS:
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"literal sub {old[:60]!r}: {n} matches, expected 1")
        text = text.replace(old, new)
        report.append(f"{n:>3}x  {old[:70]}")
    return text, report


def main() -> int:
    src = SRC.read_text(encoding="utf-8")
    t = templates()
    totals = {"bf16": load(BF_JSON)["payload_bytes"], "fp8": load(FP8_JSON)["payload_bytes"],
              "nvfp4": load(NV_JSON)["payload_bytes"]}
    key_of = {"bf16": "b16", "fp8": "b8", "nvfp4": "b4"}

    def page_total(mode: str) -> int:
        k = key_of[mode]
        lin = sum(v[k] for v in t["arch"]["lm_slide"]["items"].values()) * len(t["arch"]["lm_slide"]["indices"])
        glo = sum(v[k] for v in t["arch"]["lm_full"]["items"].values()) * len(t["arch"]["lm_full"]["indices"])
        return lin + glo + sum(v[k] for v in t["vis"].values()) + sum(v[k] for v in t["io"].values())

    for m in totals:
        if page_total(m) != totals[m]:
            print(f"GATE FAIL: page {m} = {page_total(m):,} B vs measured {totals[m]:,} B", file=sys.stderr)
            return 3
    print(f"data · {len(t['arch']['lm_slide']['indices'])} sliding {t['arch']['lm_slide']['indices']} · "
          f"{len(t['arch']['lm_full']['indices'])} global {t['arch']['lm_full']['indices']} · "
          f"vision {len(t['vis'])}x27 · io {len(t['io'])}")
    print("GATE payload == measured ✔ " + " · ".join(f"{m} {page_total(m)/1e9:.4f} GB" for m in totals))

    data_js = build_data_js(t)
    lines = src.split("\n")
    i0 = next(i for i, l in enumerate(lines) if l.startswith("const CFG = {"))
    i1 = next(i for i, l in enumerate(lines) if l.startswith("class CanvasRenderer{"))
    print(f"splice · src lines {i0+1}..{i1} ({i1-i0} lines) -> data.js ({len(data_js.splitlines())} lines)")
    text = "\n".join(lines[:i0] + data_js.split("\n") + [''] + lines[i1:])

    for anchor, repl in REWRITES:
        hits = [i for i, l in enumerate(text.split("\n")) if anchor in l]
        if len(hits) != 1:
            print(f"ANCHOR FAIL {anchor[:60]!r}: {len(hits)} lines", file=sys.stderr)
            return 4
        ls = text.split("\n")
        ls[hits[0]] = repl
        text = "\n".join(ls)
    print(f"panels · {len(REWRITES)} whole-line rewrites applied")

    text, rep = patch_code(text)
    print("code  · " + str(sum(1 for r in rep if not r.startswith("  0x"))) + " patterns matched, " +
          f"{len(rep)} attempted")

    # residue gates
    must_have = ["DiffusionGemma", "diffusiongemma-26B-A4B-it", "128 experts", "sliding",
                 "NVFP4", "FP8", "canvas", "self-conditioning"]
    must_not = ["DeepSeek", "DSpark", "Engram", "CSA2", "CSA 2", "129,280", "5,120", "384 routed",
                "mHC", "Engram tables", "wo_a", "wq_a", "Top-512", "890", "40 backbone", "40 layers",
                "engram"]
    body = text.split("</head>", 1)[-1]
    for token in must_have:
        if token.lower() not in body.lower():
            print(f"RESIDUE FAIL: '{token}' missing from the built page", file=sys.stderr)
            return 5
    bad = [tok for tok in must_not if tok.lower() in body.lower()]
    if bad:
        print(f"RESIDUE FAIL: original-model tokens survive: {bad}", file=sys.stderr)
        return 6

    # mandatory preflight (skill: tensor-atlas-v4-retarget) — runs on the candidate page BEFORE it
    # lands on disk, and proves the checker itself can still fail (--selftest)
    import subprocess
    import tempfile
    pre = DIR / "preflight.py"
    if pre.exists():
        st = subprocess.run([sys.executable, str(pre), "--selftest"], capture_output=True, text=True)
        print((st.stdout or st.stderr).strip().splitlines()[-1] if (st.stdout or st.stderr).strip() else "self-test: no output")
        if st.returncode != 0:
            print("PREFLIGHT SELF-TEST FAILED — the guard is broken, refusing to ship", file=sys.stderr)
            return 8
        with tempfile.TemporaryDirectory() as td:
            cand = Path(td) / "candidate"
            cand.mkdir()
            (cand / "index.html").write_text(text, encoding="utf-8")
            r = subprocess.run([sys.executable, str(pre), str(cand)], capture_output=True, text=True)
            print("\n".join(r.stdout.strip().splitlines()[-12:]))
            if r.returncode != 0:
                print("PREFLIGHT FAILED — index.html NOT written", file=sys.stderr)
                return 7
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.name} · {len(text):,} B")

    # the NTT internal badge + copyright footer are part of the published page
    import badge
    print("badge ·", badge.inject(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
