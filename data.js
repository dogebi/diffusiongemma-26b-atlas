// ---------- data.js ----------
/* DiffusionGemma 26B A4B — Tensor Atlas. Model facts come from the published config.json of
   google/diffusiongemma-26B-A4B-it; every byte is audited against the safetensors headers of
   three published repositories (BF16, FP8-dynamic, NVFP4), summed per tensor, per-expert
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
  rope_parameters:{sliding_attention:{rope_theta:10000.0,rope_type:'default'},
                   full_attention:{rope_theta:1000000.0,rope_type:'proportional',partial_rotary_factor:0.25}},
  bos_token_id:2,eos_token_id:1,pad_token_id:0
 },
 image_token_id:258880,canvas_length:256,
 sampling:{max_steps:48,temperature:'0.8 → 0.4 linear',entropy_bound:0.1,stop_entropy:0.005},
 vision_config:{model_type:'gemma4_vision',num_hidden_layers:27,hidden_size:1152,
  num_attention_heads:16,num_key_value_heads:16,head_dim:72,
  intermediate_size:4304,patch_size:16,default_output_length:280}
};
const CT=CFG.text_config, VC=CFG.vision_config;
const COL={blue:'#638bff',enc:'#5ca7ff',dec:'#55d7c1',engram:'#b99bff',expert:'#76b9ff',shared:'#d6e99c',attn:'#55ddd0',router:'#f4b765',mhc:'#e3a8dc',vision:'#80ceea',head:'#c4d0ff',full:'#efbc71',reindex:'#b798ff',reuse:'#709bbd',swa:'#758090',norm:'#a7b5cc',muted:'#8390a6'};
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const lerp=(a,b,t)=>a+(b-a)*t;
const smooth=x=>x*x*(3-2*x);
const num=x=>Math.round(x).toLocaleString('en-US');
function fmtP(p){return p>=1e12?(p/1e12).toFixed(3)+'T':p>=1e9?(p/1e9).toFixed(2)+'B':p>=1e6?(p/1e6).toFixed(2)+'M':p>=1e3?(p/1e3).toFixed(1)+'K':num(p);}
function bytes(n,binary=false){let b=binary?1024:1000,u=binary?['B','KiB','MiB','GiB','TiB']:['B','KB','MB','GB','TB'],k=0;while(n>=b&&k<4){n/=b;k++;}return (k===0?num(n):n.toFixed(n>=100?1:2))+' '+u[k];}
const SOURCE_BASE='https://huggingface.co/google/diffusiongemma-26B-A4B-it';
const SOURCES=[
 ['Model card',SOURCE_BASE,'Block-diffusion Gemma 4 MoE: 25.2B total / 3.8B active, 30 blocks, 8 of 128 experts per token, a 256-token canvas and 256K context. The benchmark table on this page is copied from it.'],
 ['Released config',SOURCE_BASE+'/blob/main/config.json','Text and vision configs embedded field-for-field: layer_types (5 full / 25 sliding), 128 experts top-8, global_head_dim 512, canvas_length 256, sliding_window 1024.'],
 ['BF16 checkpoint',SOURCE_BASE+'/tree/main','11 shards, 1,047 logical tensors, 51.6476 GB. Every byte on this page was read from these headers over HTTP Range; no weights were downloaded.'],
 ['FP8-dynamic build','https://huggingface.co/RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic','compressed-tensors W8A8 in a single 27.20 GB shard: e4m3 weights with per-channel scales and per-token dynamic activation scales. Its 255-entry ignore list keeps the routers, layer scalars and norms in BF16.'],
 ['NVFP4 build','https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4','modelopt NVFP4, 18.82 GB in two shards. The ignore list is the honest part: lm_head, embed_vision, *mlp*, *router*, *self_attn*, *self_conditioning* and *vision_tower* all stay BF16 — only the expert banks were converted.'],
 ['Licence',SOURCE_BASE+'/blob/main/README.md','Apache-2.0 (Gemma 4 licence). Page engine MIT, © 2026 netsin. Metadata only: no weight data is bundled in this file.']
];
const MODE_INFO={
 bf16:{label:'BF16 checkpoint',short:'BF16',color:COL.enc,note:'google/diffusiongemma-26B-A4B-it · 11 shards · 2 bytes per parameter'},
 fp8:{label:'FP8-dynamic',short:'FP8',color:COL.dec,note:'RedHatAI · compressed-tensors W8A8 · channel scales · routers and norms stay BF16'},
 nvfp4:{label:'NVFP4 experts',short:'NVFP4',color:COL.expert,note:'nvidia/modelopt · block-16 e2m1 + e4m3 scales · experts only'}
};
const MODE_KEYS=['bf16','fp8','nvfp4'];
/* per-layer memory mode: there is no KV compression in this checkpoint — the split is the
   1024-token sliding window against the five layers that see the whole context */
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
const SLIDE_W=[
  /* one sliding-window block: attention + router + 128 experts + shared MLP */
  W("experts.down_proj  (128 experts x (704 -> 2816), fused)",[128, 2816, 704],"expert","",{bf16:507510784,fp8:254476288,nvfp4:142738432}),
  W("experts.gate_up_proj  (128 experts x (gate + up), fused)",[128, 1408, 2816],"expert","",{bf16:1015021568,fp8:507871232,nvfp4:285476864}),
  W("input_layernorm.weight",[2816],"norm","",{bf16:5632,fp8:5632,nvfp4:5632}),
  W("layer_scalar  (per-layer output scale)",[1],"norm","",{bf16:2,fp8:2,nvfp4:2}),
  W("mlp.down_proj.weight  (shared dense MLP, down)",[2816, 2112],"shared","",{bf16:11894784,fp8:5953024,nvfp4:11894784}),
  W("mlp.gate_proj.weight  (shared dense MLP, gate)",[2112, 2816],"shared","",{bf16:11894784,fp8:5951616,nvfp4:11894784}),
  W("mlp.up_proj.weight  (shared dense MLP, up)",[2112, 2816],"shared","",{bf16:11894784,fp8:5951616,nvfp4:11894784}),
  W("post_attention_layernorm.weight",[2816],"norm","",{bf16:5632,fp8:5632,nvfp4:5632}),
  W("post_feedforward_layernorm.weight",[2816],"norm","",{bf16:5632,fp8:5632,nvfp4:5632}),
  W("post_feedforward_layernorm_1.weight",[2816],"norm","",{bf16:5632,fp8:5632,nvfp4:5632}),
  W("post_feedforward_layernorm_2.weight",[2816],"norm","",{bf16:5632,fp8:5632,nvfp4:5632}),
  W("pre_feedforward_layernorm.weight",[2816],"norm","",{bf16:5632,fp8:5632,nvfp4:5632}),
  W("pre_feedforward_layernorm_2.weight",[2816],"norm","",{bf16:5632,fp8:5632,nvfp4:5632}),
  W("router.per_expert_scale  (per-expert routing scale)",[128],"router","",{bf16:256,fp8:256,nvfp4:256}),
  W("router.proj.weight  (router: 128 logits per token)",[128, 2816],"router","",{bf16:720896,fp8:720896,nvfp4:720896}),
  W("router.scale  (pre-router scale)",[2816],"router","",{bf16:5632,fp8:5632,nvfp4:5632}),
  W("self_attn.k_norm.weight  (per-head K norm)",[256],"attn","",{bf16:512,fp8:512,nvfp4:512}),
  W("self_attn.k_proj.weight  (K projection (GQA))",[2048, 2816],"attn","",{bf16:11534336,fp8:5771264,nvfp4:11534336}),
  W("self_attn.o_proj.weight  (attention output)",[2816, 4096],"attn","",{bf16:23068672,fp8:11539968,nvfp4:23068672}),
  W("self_attn.q_norm.weight  (per-head Q norm)",[256],"attn","",{bf16:512,fp8:512,nvfp4:512}),
  W("self_attn.q_proj.weight  (Q projection)",[4096, 2816],"attn","",{bf16:23068672,fp8:11542528,nvfp4:23068672}),
  W("self_attn.v_proj.weight  (V projection (GQA))",[2048, 2816],"attn","",{bf16:11534336,fp8:5771264,nvfp4:11534336}),
];
const FULL_W=[
  /* one global block: q/k/o (no v_proj), same router and expert banks */
  W("experts.down_proj  (128 experts x (704 -> 2816), fused)",[128, 2816, 704],"expert","",{bf16:507510784,fp8:254476288,nvfp4:142738432}),
  W("experts.gate_up_proj  (128 experts x (gate + up), fused)",[128, 1408, 2816],"expert","",{bf16:1015021568,fp8:507871232,nvfp4:285476864}),
  W("input_layernorm.weight",[2816],"norm","",{bf16:5632,fp8:5632,nvfp4:5632}),
  W("layer_scalar  (per-layer output scale)",[1],"norm","",{bf16:2,fp8:2,nvfp4:2}),
  W("mlp.down_proj.weight  (shared dense MLP, down)",[2816, 2112],"shared","",{bf16:11894784,fp8:5953024,nvfp4:11894784}),
  W("mlp.gate_proj.weight  (shared dense MLP, gate)",[2112, 2816],"shared","",{bf16:11894784,fp8:5951616,nvfp4:11894784}),
  W("mlp.up_proj.weight  (shared dense MLP, up)",[2112, 2816],"shared","",{bf16:11894784,fp8:5951616,nvfp4:11894784}),
  W("post_attention_layernorm.weight",[2816],"norm","",{bf16:5632,fp8:5632,nvfp4:5632}),
  W("post_feedforward_layernorm.weight",[2816],"norm","",{bf16:5632,fp8:5632,nvfp4:5632}),
  W("post_feedforward_layernorm_1.weight",[2816],"norm","",{bf16:5632,fp8:5632,nvfp4:5632}),
  W("post_feedforward_layernorm_2.weight",[2816],"norm","",{bf16:5632,fp8:5632,nvfp4:5632}),
  W("pre_feedforward_layernorm.weight",[2816],"norm","",{bf16:5632,fp8:5632,nvfp4:5632}),
  W("pre_feedforward_layernorm_2.weight",[2816],"norm","",{bf16:5632,fp8:5632,nvfp4:5632}),
  W("router.per_expert_scale  (per-expert routing scale)",[128],"router","",{bf16:256,fp8:256,nvfp4:256}),
  W("router.proj.weight  (router: 128 logits per token)",[128, 2816],"router","",{bf16:720896,fp8:720896,nvfp4:720896}),
  W("router.scale  (pre-router scale)",[2816],"router","",{bf16:5632,fp8:5632,nvfp4:5632}),
  W("self_attn.k_norm.weight  (per-head K norm)",[512],"attn","",{bf16:1024,fp8:1024,nvfp4:1024}),
  W("self_attn.k_proj.weight  (K projection (GQA))",[1024, 2816],"attn","",{bf16:5767168,fp8:2885632,nvfp4:5767168}),
  W("self_attn.o_proj.weight  (attention output)",[2816, 8192],"attn","",{bf16:46137344,fp8:23074304,nvfp4:46137344}),
  W("self_attn.q_norm.weight  (per-head Q norm)",[512],"attn","",{bf16:1024,fp8:1024,nvfp4:1024}),
  W("self_attn.q_proj.weight  (Q projection)",[8192, 2816],"attn","",{bf16:46137344,fp8:23085056,nvfp4:46137344}),
];
const EMBED_W=[
  /* the tied embedding / unembedding */
  W("model.decoder.embed_tokens.weight",[262144, 2816],"vocab","",{bf16:1476395008,fp8:1476395008,nvfp4:1476395008}),
];
const HEAD_W=[
  /* final RMSNorm — the head itself is tied */
  W("model.decoder.norm.weight",[2816],"head","",{bf16:5632,fp8:5632,nvfp4:5632}),
];
const SELFCOND_W=[
  /* diffusion self-conditioning */
  W("model.decoder.self_conditioning.down_proj.weight",[2816, 2112],"selfcond","",{bf16:11894784,fp8:11894784,nvfp4:11894784}),
  W("model.decoder.self_conditioning.gate_proj.weight",[2112, 2816],"selfcond","",{bf16:11894784,fp8:11894784,nvfp4:11894784}),
  W("model.decoder.self_conditioning.pre_norm.weight",[2816],"selfcond","",{bf16:5632,fp8:5632,nvfp4:5632}),
  W("model.decoder.self_conditioning.up_proj.weight",[2112, 2816],"selfcond","",{bf16:11894784,fp8:11894784,nvfp4:11894784}),
];
const VISION_W=[
  /* one vision block, x27 (bytes already summed) */
  W("model.encoder.vision_tower.encoder.layers.*.input_layernorm.weight",[1152],"vision","",{bf16:62208,fp8:62208,nvfp4:62208},27),
  W("model.encoder.vision_tower.encoder.layers.*.mlp.down_proj.linear.weight",[1152, 4304],"vision","",{bf16:267743232,fp8:267743232,nvfp4:267743232},27),
  W("model.encoder.vision_tower.encoder.layers.*.mlp.gate_proj.linear.weight",[4304, 1152],"vision","",{bf16:267743232,fp8:267743232,nvfp4:267743232},27),
  W("model.encoder.vision_tower.encoder.layers.*.mlp.up_proj.linear.weight",[4304, 1152],"vision","",{bf16:267743232,fp8:267743232,nvfp4:267743232},27),
  W("model.encoder.vision_tower.encoder.layers.*.post_attention_layernorm.weight",[1152],"vision","",{bf16:62208,fp8:62208,nvfp4:62208},27),
  W("model.encoder.vision_tower.encoder.layers.*.post_feedforward_layernorm.weight",[1152],"vision","",{bf16:62208,fp8:62208,nvfp4:62208},27),
  W("model.encoder.vision_tower.encoder.layers.*.pre_feedforward_layernorm.weight",[1152],"vision","",{bf16:62208,fp8:62208,nvfp4:62208},27),
  W("model.encoder.vision_tower.encoder.layers.*.self_attn.k_norm.weight  (ViT per-head K norm)",[72],"vision","",{bf16:3888,fp8:3888,nvfp4:3888},27),
  W("model.encoder.vision_tower.encoder.layers.*.self_attn.k_proj.linear.weight",[1152, 1152],"vision","",{bf16:71663616,fp8:71663616,nvfp4:71663616},27),
  W("model.encoder.vision_tower.encoder.layers.*.self_attn.o_proj.linear.weight",[1152, 1152],"vision","",{bf16:71663616,fp8:71663616,nvfp4:71663616},27),
  W("model.encoder.vision_tower.encoder.layers.*.self_attn.q_norm.weight  (ViT per-head Q norm)",[72],"vision","",{bf16:3888,fp8:3888,nvfp4:3888},27),
  W("model.encoder.vision_tower.encoder.layers.*.self_attn.q_proj.linear.weight",[1152, 1152],"vision","",{bf16:71663616,fp8:71663616,nvfp4:71663616},27),
  W("model.encoder.vision_tower.encoder.layers.*.self_attn.v_proj.linear.weight",[1152, 1152],"vision","",{bf16:71663616,fp8:71663616,nvfp4:71663616},27),
];
const ENC_W=[
  /* encoder-side: patch embedder, position table, std vectors, vision projector, 30 layer scales */
  W("model.encoder.embed_vision.embedding_projection.weight",[2816, 1152],"vision","",{bf16:6488064,fp8:6488064,nvfp4:6488064}),
  W("model.encoder.language_model.layers.0.layer_scalar",[1],"norm","",{bf16:2,fp8:2,nvfp4:2}),
  W("model.encoder.language_model.layers.1.layer_scalar",[1],"norm","",{bf16:2,fp8:2,nvfp4:2}),
  W("model.encoder.language_model.layers.10.layer_scalar",[1],"norm","",{bf16:2,fp8:2,nvfp4:2}),
  W("model.encoder.language_model.layers.11.layer_scalar",[1],"norm","",{bf16:2,fp8:2,nvfp4:2}),
  W("model.encoder.language_model.layers.12.layer_scalar",[1],"norm","",{bf16:2,fp8:2,nvfp4:2}),
  W("model.encoder.language_model.layers.13.layer_scalar",[1],"norm","",{bf16:2,fp8:2,nvfp4:2}),
  W("model.encoder.language_model.layers.14.layer_scalar",[1],"norm","",{bf16:2,fp8:2,nvfp4:2}),
  W("model.encoder.language_model.layers.15.layer_scalar",[1],"norm","",{bf16:2,fp8:2,nvfp4:2}),
  W("model.encoder.language_model.layers.16.layer_scalar",[1],"norm","",{bf16:2,fp8:2,nvfp4:2}),
  W("model.encoder.language_model.layers.17.layer_scalar",[1],"norm","",{bf16:2,fp8:2,nvfp4:2}),
  W("model.encoder.language_model.layers.18.layer_scalar",[1],"norm","",{bf16:2,fp8:2,nvfp4:2}),
  W("model.encoder.language_model.layers.19.layer_scalar",[1],"norm","",{bf16:2,fp8:2,nvfp4:2}),
  W("model.encoder.language_model.layers.2.layer_scalar",[1],"norm","",{bf16:2,fp8:2,nvfp4:2}),
  W("model.encoder.language_model.layers.20.layer_scalar",[1],"norm","",{bf16:2,fp8:2,nvfp4:2}),
  W("model.encoder.language_model.layers.21.layer_scalar",[1],"norm","",{bf16:2,fp8:2,nvfp4:2}),
  W("model.encoder.language_model.layers.22.layer_scalar",[1],"norm","",{bf16:2,fp8:2,nvfp4:2}),
  W("model.encoder.language_model.layers.23.layer_scalar",[1],"norm","",{bf16:2,fp8:2,nvfp4:2}),
  W("model.encoder.language_model.layers.24.layer_scalar",[1],"norm","",{bf16:2,fp8:2,nvfp4:2}),
  W("model.encoder.language_model.layers.25.layer_scalar",[1],"norm","",{bf16:2,fp8:2,nvfp4:2}),
  W("model.encoder.language_model.layers.26.layer_scalar",[1],"norm","",{bf16:2,fp8:2,nvfp4:2}),
  W("model.encoder.language_model.layers.27.layer_scalar",[1],"norm","",{bf16:2,fp8:2,nvfp4:2}),
  W("model.encoder.language_model.layers.28.layer_scalar",[1],"norm","",{bf16:2,fp8:2,nvfp4:2}),
  W("model.encoder.language_model.layers.29.layer_scalar",[1],"norm","",{bf16:2,fp8:2,nvfp4:2}),
  W("model.encoder.language_model.layers.3.layer_scalar",[1],"norm","",{bf16:2,fp8:2,nvfp4:2}),
  W("model.encoder.language_model.layers.4.layer_scalar",[1],"norm","",{bf16:2,fp8:2,nvfp4:2}),
  W("model.encoder.language_model.layers.5.layer_scalar",[1],"norm","",{bf16:2,fp8:2,nvfp4:2}),
  W("model.encoder.language_model.layers.6.layer_scalar",[1],"norm","",{bf16:2,fp8:2,nvfp4:2}),
  W("model.encoder.language_model.layers.7.layer_scalar",[1],"norm","",{bf16:2,fp8:2,nvfp4:2}),
  W("model.encoder.language_model.layers.8.layer_scalar",[1],"norm","",{bf16:2,fp8:2,nvfp4:2}),
  W("model.encoder.language_model.layers.9.layer_scalar",[1],"norm","",{bf16:2,fp8:2,nvfp4:2}),
  W("model.encoder.vision_tower.patch_embedder.input_proj.weight",[1152, 768],"vision","",{bf16:1769472,fp8:1769472,nvfp4:1769472}),
  W("model.encoder.vision_tower.patch_embedder.position_embedding_table",[2, 10240, 1152],"vision","",{bf16:47185920,fp8:47185920,nvfp4:47185920}),
  W("model.encoder.vision_tower.std_bias",[1152],"vision","",{bf16:2304,fp8:2304,nvfp4:2304}),
  W("model.encoder.vision_tower.std_scale",[1152],"vision","",{bf16:2304,fp8:2304,nvfp4:2304}),
];
function layerWeights(i){return modeFor(i)==='full'?FULL_W:SLIDE_W;}
const LAYERS=Array.from({length:30},(_,i)=>({id:'L'+i,index:i,label:'Layer '+String(i).padStart(2,'0'),
 part:modeFor(i)==='full'?'global':'sliding',mode:modeFor(i),owner:ownerFor(i),indexOwner:indexOwnerFor(i),
 ratio:1,ws:layerWeights(i)}));
const ENGRAM=[];                       /* no memory tables in this checkpoint */
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
const MOE_EXPERT_P=(2*2816*704+704*2816)/1e6;
const MOE_TOTAL_P=MOE_EXPERT_P*128*30;
const NV_DELTA=TOTALS.bf16-TOTALS.nvfp4;
const FP8_DELTA=TOTALS.bf16-TOTALS.fp8;
const CATEGORIES=[
 ['expert','Routed experts · 128 per layer',COL.expert,LAYERS.flatMap(l=>l.ws.filter(w=>w.cat==='expert'))],
 ['attn','Attention',COL.attn,LAYERS.flatMap(l=>l.ws.filter(w=>w.cat==='attn'))],
 ['shared','Shared dense MLP',COL.shared,LAYERS.flatMap(l=>l.ws.filter(w=>w.cat==='shared'))],
 ['router','Routers and scales',COL.router,LAYERS.flatMap(l=>l.ws.filter(w=>w.cat==='router'))],
 ['vision','Vision tower + projector',COL.vision,[...VISION.ws,...ALIGNER.ws]],
 ['selfcond','Self-conditioning',COL.engram,SELF.ws],
 ['vocab','Tied embedding',COL.head,EMBED.ws],
 ['norm','Norms and layer scales',COL.norm,ALL_W.filter(w=>w.cat==='norm')]
];
const EXP={
 overview:{title:'Two passes, one set of blocks.',body:'DiffusionGemma is a Gemma 4 Mixture-of-Experts that generates text by discrete diffusion instead of token-by-token sampling. The checkpoint holds ONE set of 30 blocks: the encoder runs them causally over the prompt to build the KV cache, then the decoder runs them bidirectionally over a 256-token canvas, denoising it in parallel steps. Weights are shared; the two roles are not. 45.7 GB of the 51.6 GB checkpoint is the expert banks, which is why the quantized builds all attack the same 88%.'},
 expert:{title:'128 experts, 8 per token, 88% of the file.',body:'Each block stores its experts fused: gate_up_proj [128, 1408, 2816] and down_proj [128, 2816, 704] — one tensor per bank, with the 128 experts stacked inside. That is 1015.0 MB + 507.5 MB per block, 45.7 GB across 30 blocks: 88.4% of the BF16 checkpoint. All 128 experts stay on disk for every token; the router picks 8 and mixes them by weight, beside an always-on shared MLP with the same width budget (2112 intermediate).'},
 full:{title:'The five global blocks.',body:'Layers 5, 11, 17, 23 and 29 leave the sliding window and attend the whole context: 16 query heads over 2 global KV heads at head_dim 512, so k is 1024 wide and q/o are 8192 wide. There is no separate v_proj in these five blocks — only q, k and o are in the checkpoint. Their rotary position covers a quarter of each head (partial_rotary_factor 0.25, theta 1e6), while the other twenty-five blocks use theta 1e4.'},
 swa:{title:'The twenty-five sliding blocks.',body:'A 1024-token window: 16 query heads over 8 KV heads at head_dim 256, with per-head Q and K norms. Their KV cache is capped at 1024 positions, so a 256K conversation does not grow them at all — only the five global blocks grow. Attention is under 4% of a sliding block; the fused expert banks are 1.52 GB of it at BF16.'},
 selfcond:{title:'Self-conditioning: 35.7 MB that make the decoder a diffusion model.',body:'The decoder is not autoregressive. It denoises all 256 canvas positions at once and can feed its previous estimate back in — this module is what does that: a pre-norm plus gate/up/down projections (2816 → 2112 → 2816). The encoder pass never calls it, which is also why it is one of the modules the NVFP4 build leaves in BF16.'},
 vision:{title:'A 27-block vision tower and a 6.5 MB door.',body:'Images are patchified at 16×16 and run through 27 blocks of hidden 1152 — 16 heads at head_dim 72 and a 4304-wide MLP — then projected by embed_vision [2816, 1152] into the language hidden size. The tower carries its own patch embedder, a [2, 10240, 1152] position table and two 1152-wide correction vectors. 1.1 GB of the checkpoint, BF16 in all three builds.'},
 bf16:{title:'The reference checkpoint.',body:'51.6476 GB in 11 shards: 45.7 GB of routed experts, 1.48 GB of tied embedding, 1.06 GB of attention, ~1.1 GB of vision tower, 0.36 GB of shared MLPs, 36 MB of self-conditioning and the rest norms and layer scales.'},
 fp8:{title:'FP8-dynamic: everything except the routers, norms and embedding.',body:'27.1977 GB in a single shard — 1.90× smaller. Every linear weight becomes e4m3 with a per-channel scale, activations get per-token dynamic scales, and the experts are included. The 255-entry ignore list keeps routers, layer_scalars and RMSNorms in BF16, and the 1.48 GB embedding is untouched too.'},
 nvfp4:{title:'NVFP4: only the experts move.',body:'18.8181 GB in two shards — 2.74× smaller than BF16. The reason it is not smaller is the ignore list: lm_head, embed_vision, *mlp*, *router*, *self_attn*, *self_conditioning* and *vision_tower* all stay BF16. Only the fused expert banks are block-16 e2m1 with e4m3 scales, and each expert also carries a per-tensor F32 scale. The tied embedding alone is 1.48 GB of these 18.8 GB.'},
 storage:{title:'Where 51.6 GB goes.',body:'88.4% is the routed experts. Flip the precision and watch which categories move: NVFP4 converts exactly one of them, FP8 converts everything except the embedding, the routers and the norms.'},
 cache:{title:'Two attention behaviours in one checkpoint.',body:'Twenty-five blocks see 1024 tokens; five see everything. At 256K context the sliding blocks cost exactly what they cost at 1K, and only the five global blocks keep growing — 2 KV heads at head_dim 512 each.'}
};
const BENCH=[
 {name:'MMLU Pro',capability:'Knowledge',values:[77.6,82.6]},
 {name:'AIME 2026 · no tools',capability:'Math',values:[69.1,88.3]},
 {name:'LiveCodeBench v6',capability:'Coding',values:[69.1,77.1]},
 {name:'GPQA Diamond',capability:'Science',values:[73.2,82.3]},
 {name:'Tau2 · average of 3',capability:'Agentic tools',values:[56.2,68.2]},
 {name:'HLE · no tools',capability:'Reasoning',values:[11.0,8.7]},
 {name:'HLE · with search',capability:'Reasoning',values:[11.9,17.2]},
 {name:'BigBench Extra Hard',capability:'Reasoning',values:[47.6,64.8]},
 {name:'MMMLU',capability:'Multilingual',values:[81.5,86.3]},
 {name:'MMMU Pro',capability:'Vision',values:[54.3,73.8]},
 {name:'MATH-Vision',capability:'Vision math',values:[70.5,82.4]},
 {name:'MedXPertQA MM',capability:'Vision knowledge',values:[49.0,58.1]},
 {name:'MRCR v2 · 8-needle 128k',capability:'Long context',values:[32.0,44.1]}
];
const BENCH_MODELS=['DiffusionGemma 26B A4B','Gemma 4 26B A4B'];
function pickExperts(seed,n=128,k=8){let x=(seed+1)*2654435761>>>0;const s=new Set;while(s.size<k){x=(Math.imul(x,1664525)+1013904223)>>>0;s.add(x%n);}return [...s].sort((a,b)=>a-b);}
const PHASES=[
 {name:'Encode',label:'Encoder',from:0,to:6,active:'prefill',color:COL.enc,caption:'Read the prompt, build the KV cache.',desc:'The 30 blocks run causally over the prompt. Every token routes to 8 of 128 experts; the 25 sliding blocks keep a bounded 1024-token window and the 5 global blocks keep everything. Images enter the same stream through the vision tower and embed_vision.'},
 {name:'Canvas',label:'Canvas',from:6,to:11,active:'256 positions',color:COL.dec,caption:'Start from noise: 256 masked positions.',desc:'The decoder does not generate one token at a time. All 256 canvas positions start masked and are denoised in parallel with bidirectional attention over the canvas plus access to the cached prompt.'},
 {name:'Denoise',label:'Denoise',from:11,to:17,active:'≤48 steps',color:COL.expert,caption:'Iterate until the canvas is confident.',desc:'Each step re-predicts every position. The sampler keeps the lowest-entropy tokens whose mutual-information bound stays under 0.1, renoises the rest, and stops early when the average canvas entropy falls below 0.005 and two consecutive steps agree. Self-conditioning feeds the previous estimate into the next step.'},
 {name:'Append',label:'Append',from:17,to:20,active:'encoder again',color:COL.shared,caption:'The accepted canvas joins the context.',desc:'A finished canvas is appended to the KV cache and the encoder pass runs again — which is why one weight set serves both directions: the two stacks are the same 30 blocks.'},
 {name:'Repeat',label:'Repeat',from:20,to:22,active:'15-20 tok/pass',color:COL.full,caption:'Many tokens per forward pass.',desc:'Because a whole canvas is denoised at once, the model emits 15-20 tokens per forward pass, which is where the reported 1100+ tokens per second at low batch size comes from.'}
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
 if(n.includes('input_layernorm'))return 14; if(n.includes('post_attention'))return 15; if(n.includes('feedforward'))return 16; if(n.includes('layer_scalar'))return 17;
 if(n.includes('self_conditioning'))return 18; if(n.includes('embed_tokens'))return 0;
 if(n.includes('patch_embedder'))return 0; if(n.includes('std_'))return 1; if(n.includes('embed_vision'))return 2;
 if(n.includes('vision_tower.encoder'))return 3; return 19;};
 return [...ws].sort((a,b)=>rank(a)-rank(b));}
function isAttentionTensor(w){return w.cat==='attn';}
function tensorInfo(w,m){
 const p=fmtP(w.p),size=bytes(wBytes(w,'bf16'));
 let t='Stored tensor.',b='Two bytes per parameter in the BF16 checkpoint. The storage view shows what each quantized build does with it.';
 if(w.name.includes('experts.gate_up')){t='Fused expert gate and up banks.';b='One tensor holds all 128 experts: [128, 1408, 2816]. Inside each expert, gate (704 x 2816) and up (704 x 2816) sit side by side and the block computes silu(gate) x up. 1015.0 MB per block at BF16 and the largest tensor class in the checkpoint — 30 of them are 30.5 GB.';}
 else if(w.name.includes('experts.down')){t='Fused expert down bank.';b='[128, 2816, 704]: each expert projects its 704-wide activation back to the 2816-dim residual stream. 507.5 MB per block at BF16, 15.2 GB across the stack.';}
 else if(w.name.includes('router.proj')){t='Router.';b='2816 -> 128 logits, one per expert. The top 8 fire, their outputs are mixed by the routing weights, and the always-on shared MLP runs beside them. 0.72 MB per block, and both quantized builds refuse to touch it: a misrouting router costs more than an 8-bit one.';}
 else if(w.name.includes('router.per_expert_scale')){t='Per-expert routing scale.';b='128 learned scalars multiplying each expert routing weight before the top-k selection.';}
 else if(w.name.endsWith('router.scale')){t='Pre-router scale.';b='2816 learned scales applied to the hidden state before routing — Gemma-style decoration of the router input.';}
 else if(w.name.includes('mlp.')){t='Shared dense MLP.';b='The always-on path beside the experts: 2816 -> 2112 -> 2816 with gelu-tanh. Every token passes through it, so it is the one part of the feed-forward with no routing and no sparsity; 11.9 MB per matrix.';}
 else if(w.name.includes('self_conditioning')){t='Self-conditioning projection.';b='2816 -> 2112 -> 2816 plus a pre-norm, applied to the decoder previous estimate before the next denoising step. 35.7 MB in total and unique to the diffusion decoder.';}
 else if(w.name.includes('q_proj')){t='Query projection.';b='The five global blocks project to 8192 (16 heads x 512); the twenty-five sliding blocks project to 4096 (16 heads x 256). Per-head Q norm follows in both.';}
 else if(w.name.includes('k_proj')){t='Key projection.';b='Global blocks: 2 KV heads x 512 = 1024 rows. Sliding blocks: 8 KV heads x 256 = 2048. Grouped-query attention, with per-head K norm.';}
 else if(w.name.includes('v_proj')){t='Value projection.';b='Present in the twenty-five sliding blocks only: 8 KV heads x 256 = 2048 rows. The five global blocks store no separate v_proj.';}
 else if(w.name.includes('o_proj')){t='Attention output projection.';b='4096 -> 2816 in the sliding blocks, 8192 -> 2816 in the global ones.';}
 else if(w.name.includes('q_norm')||w.name.includes('k_norm')){t='Per-head norm.';b='256 scales in the sliding blocks, 512 in the global ones: RMSNorm applied per head to queries and keys before the dot product.';}
 else if(w.name.includes('layer_scalar')){t='Layer scalar.';b='A single learned scalar multiplying the block output. It appears once per block per stack — the encoder side and the decoder side carry separate layer scales over the same weights, which is the clearest evidence that one weight set serves both passes.';}
 else if(w.name.includes('embed_tokens')){t='Token embedding — and the unembedding too.';b='[262144, 2816]: 1.48 GB, the whole vocabulary, tied (tie_word_embeddings true), which is why the checkpoint has no lm_head tensor at all. Both quantized builds keep it BF16.';}
 else if(w.name.includes('patch_embedder')){t='Patch embedder and position table.';b='16x16 patches, plus a [2, 10240, 1152] learned position table — the vision tower keeps its own position signal, separate from the language model rotary.';}
 else if(w.name.includes('embed_vision')){t='Vision-to-language projector.';b='[2816, 1152]: the 6.5 MB door between the 27-block vision tower and the language hidden size.';}
 else if(w.name.includes('std_bias')||w.name.includes('std_scale')){t='Vision output correction.';b='Two 1152-wide vectors rescaling the tower output before projection.';}
 else if(w.name.includes('vision_tower')){t='Vision tower tensor.';b='Part of the 27-block ViT: attention (16 heads at head_dim 72, fused qkv per block in the published layout as separate q/k/v here) or the 4304-wide GELU-tanh MLP. BF16 in all three builds.';}
 else if(w.name.includes('norm')){t='RMSNorm.';b='Layer normalisation on the residual stream (eps 1e-6). Kept BF16 by both quantized builds.';}
 return {title:t,body:b,size,p};}
function layerStory(m){
 if(!m.mode)return null;let t,body,detail;
 if(m.mode==='swa'){t='Sliding-window block';body='1024-token attention on 8 KV heads, then 8 of 128 experts plus the shared MLP.';detail='This is 25 of the 30 blocks. Its KV cache is capped at 1024 positions, so long context costs it nothing. The fused expert banks are 1.52 GB of this block at BF16; attention is under 4%.';}
 else {t='Global block';body='Full-context attention on 2 global KV heads at head_dim 512, then the same 8-of-128 expert bank.';detail='Five blocks — 5, 11, 17, 23, 29 — keep a KV cache that grows with the context. They are also the only blocks whose attention shapes differ: q/o 8192 wide, k 1024, and no separate v_proj in the checkpoint.';}
 return {title:t,body,detail};}
