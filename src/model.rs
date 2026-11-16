//! Llama-architecture transformer: config + safetensors loading + single-token forward pass.

use crate::kernels::{dot, Matrix};
use rayon::prelude::*;
use serde_json::Value;
use std::{collections::HashMap, fs, path::Path};

pub struct Config {
    pub dim: usize,
    pub hidden: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab: usize,
    pub eps: f32,
    pub rope_theta: f64,
    pub eos: Vec<u32>,
}

impl Config {
    fn parse(v: &Value) -> Result<Self, String> {
        let get = |k: &str| {
            v[k].as_u64()
                .map(|x| x as usize)
                .ok_or_else(|| format!("config.json: missing {k}"))
        };
        if v["architectures"][0] != "LlamaForCausalLM" {
            return Err(format!("unsupported architecture {}", v["architectures"]));
        }
        // ponytail: Llama 3.x "llama3" rope scaling and attention/MLP biases are not implemented yet.
        if !v["rope_scaling"].is_null() || v["attention_bias"] == true || v["mlp_bias"] == true {
            return Err("rope_scaling / bias variants are not supported yet".into());
        }
        let (dim, n_heads) = (get("hidden_size")?, get("num_attention_heads")?);
        let eos = match &v["eos_token_id"] {
            Value::Array(a) => a
                .iter()
                .filter_map(Value::as_u64)
                .map(|x| x as u32)
                .collect(),
            x => x.as_u64().map(|x| vec![x as u32]).unwrap_or_default(),
        };
        Ok(Config {
            dim,
            hidden: get("intermediate_size")?,
            n_layers: get("num_hidden_layers")?,
            n_heads,
            n_kv_heads: get("num_key_value_heads").unwrap_or(n_heads),
            head_dim: get("head_dim").unwrap_or(dim / n_heads),
            vocab: get("vocab_size")?,
            eps: v["rms_norm_eps"].as_f64().unwrap_or(1e-5) as f32,
            rope_theta: v["rope_theta"].as_f64().unwrap_or(10000.0),
            eos,
        })
    }
}

/// Raw safetensors files plus an index of where each tensor lives.
/// Tensors are converted to f32 one at a time so peak memory stays ~file size + model size.
struct Tensors {
    files: Vec<Vec<u8>>,
    index: HashMap<String, (usize, String, Vec<usize>, usize, usize)>, // file, dtype, shape, start, end
}

impl Tensors {
    fn open(dir: &Path) -> Result<Self, String> {
        let mut paths: Vec<_> = fs::read_dir(dir)
            .map_err(|e| format!("{}: {e}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        paths.sort();
        if paths.is_empty() {
            return Err(format!("no .safetensors files in {}", dir.display()));
        }
        let (mut files, mut index) = (Vec::new(), HashMap::new());
        for path in paths {
            let buf = fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            let bad = || format!("{}: malformed safetensors header", path.display());
            let n = u64::from_le_bytes(buf.get(..8).ok_or_else(bad)?.try_into().unwrap()) as usize;
            let header: Value =
                serde_json::from_slice(buf.get(8..8 + n).ok_or_else(bad)?).map_err(|_| bad())?;
            for (name, t) in header.as_object().ok_or_else(bad)? {
                if name == "__metadata__" {
                    continue;
                }
                let shape = t["shape"]
                    .as_array()
                    .ok_or_else(bad)?
                    .iter()
                    .filter_map(Value::as_u64)
                    .map(|x| x as usize)
                    .collect();
                let off = |i: usize| {
                    t["data_offsets"][i]
                        .as_u64()
                        .map(|x| 8 + n + x as usize)
                        .ok_or_else(bad)
                };
                let (start, end) = (off(0)?, off(1)?);
                if start > end || end > buf.len() {
                    return Err(bad());
                }
                let dtype = t["dtype"].as_str().unwrap_or("").to_string();
                index.insert(name.clone(), (files.len(), dtype, shape, start, end));
            }
            files.push(buf);
        }
        Ok(Tensors { files, index })
    }

    fn get(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
        let (f, dtype, shape, s, e) = self
            .index
            .get(name)
            .ok_or_else(|| format!("missing tensor {name}"))?;
        let bytes = &self.files[*f][*s..*e];
        let data: Vec<f32> = match dtype.as_str() {
            // bf16 is the top half of an f32, so widening is a shift.
            "BF16" => bytes
                .chunks_exact(2)
                .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
                .collect(),
            "F32" => bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            "F16" => bytes
                .chunks_exact(2)
                .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect(),
            d => return Err(format!("{name}: unsupported dtype {d}")),
        };
        if data.len() != shape.iter().product::<usize>() {
            return Err(format!("{name}: shape {shape:?} does not match data"));
        }
        Ok((data, shape.clone()))
    }

    fn vector(&self, name: &str, len: usize) -> Result<Vec<f32>, String> {
        let (v, _) = self.get(name)?;
        (v.len() == len)
            .then_some(v)
            .ok_or_else(|| format!("{name}: expected length {len}"))
    }

    fn matrix(&self, name: &str, rows: usize, cols: usize, q8: bool) -> Result<Matrix, String> {
        self.stacked(&[(name.to_string(), rows)], cols, q8)
    }

    /// Several matrices with the same input stacked row-wise into one, so they run as a
    /// single matvec (one parallel dispatch instead of several small ones).
    fn stacked(&self, parts: &[(String, usize)], cols: usize, q8: bool) -> Result<Matrix, String> {
        let mut data = Vec::new();
        for (name, rows) in parts {
            let (v, shape) = self.get(name)?;
            if shape != [*rows, cols] {
                return Err(format!("{name}: expected [{rows}, {cols}], got {shape:?}"));
            }
            data.extend(v);
        }
        let m = Matrix::new(data, parts.iter().map(|p| p.1).sum(), cols);
        Ok(if q8 { m.quantize() } else { m })
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let (sign, exp, frac) = (
        (h >> 15) as u32,
        ((h >> 10) & 0x1f) as u32,
        (h & 0x3ff) as u32,
    );
    let mag = match exp {
        0 => frac as f32 * 2f32.powi(-24), // subnormal
        31 => {
            if frac == 0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => f32::from_bits(((exp + 112) << 23) | (frac << 13)),
    };
    if sign == 1 {
        -mag
    } else {
        mag
    }
}

struct Layer {
    attn_norm: Vec<f32>,
    wqkv: Matrix, // q, k, v projections stacked
    wo: Matrix,
    ffn_norm: Vec<f32>,
    w_gate_up: Matrix, // gate, up projections stacked
    w_down: Matrix,
}

pub struct Model {
    pub cfg: Config,
    embed: Vec<f32>,
    layers: Vec<Layer>,
    norm: Vec<f32>,
    lm_head: Matrix,
}

impl Model {
    /// Load `config.json` + `*.safetensors` from a Hugging Face model directory.
    /// `q8` quantizes every linear layer to int8 at load time.
    pub fn load(dir: &Path, q8: bool) -> Result<Self, String> {
        let cfg_text =
            fs::read_to_string(dir.join("config.json")).map_err(|e| format!("config.json: {e}"))?;
        let cfg = Config::parse(
            &serde_json::from_str(&cfg_text).map_err(|e| format!("config.json: {e}"))?,
        )?;
        let t = Tensors::open(dir)?;
        let (d, hd) = (cfg.dim, cfg.head_dim);
        let (q_dim, kv_dim) = (cfg.n_heads * hd, cfg.n_kv_heads * hd);

        let layers = (0..cfg.n_layers)
            .map(|i| {
                let p = format!("model.layers.{i}.");
                Ok(Layer {
                    attn_norm: t.vector(&format!("{p}input_layernorm.weight"), d)?,
                    wqkv: t.stacked(
                        &[
                            (format!("{p}self_attn.q_proj.weight"), q_dim),
                            (format!("{p}self_attn.k_proj.weight"), kv_dim),
                            (format!("{p}self_attn.v_proj.weight"), kv_dim),
                        ],
                        d,
                        q8,
                    )?,
                    wo: t.matrix(&format!("{p}self_attn.o_proj.weight"), d, q_dim, q8)?,
                    ffn_norm: t.vector(&format!("{p}post_attention_layernorm.weight"), d)?,
                    w_gate_up: t.stacked(
                        &[
                            (format!("{p}mlp.gate_proj.weight"), cfg.hidden),
                            (format!("{p}mlp.up_proj.weight"), cfg.hidden),
                        ],
                        d,
                        q8,
                    )?,
                    w_down: t.matrix(&format!("{p}mlp.down_proj.weight"), d, cfg.hidden, q8)?,
                })
            })
            .collect::<Result<_, String>>()?;

        let embed = t.vector("model.embed_tokens.weight", cfg.vocab * d)?;
        // Tied embeddings: the output projection reuses the embedding table.
        // ponytail: f32 mode keeps two copies of the table; share it if memory gets tight.
        let lm_head = match t.index.contains_key("lm_head.weight") {
            true => t.matrix("lm_head.weight", cfg.vocab, d, q8)?,
            false => {
                let m = Matrix::new(embed.clone(), cfg.vocab, d);
                if q8 {
                    m.quantize()
                } else {
                    m
                }
            }
        };
        Ok(Model {
            norm: t.vector("model.norm.weight", d)?,
            cfg,
            embed,
            layers,
            lm_head,
        })
    }

    pub fn is_quantized(&self) -> bool {
        self.lm_head.is_quantized()
    }
}

/// Per-sequence state: KV cache and scratch buffers, allocated once.
pub struct Session<'a> {
    m: &'a Model,
    pub pos: usize,
    pub max_seq: usize,
    x: Vec<f32>,
    xb: Vec<f32>,
    xb2: Vec<f32>,
    qkv: Vec<f32>,
    attn_out: Vec<f32>,
    att: Vec<f32>,
    hb: Vec<f32>, // gate | up
    rope: Vec<(f32, f32)>,
    k_cache: Vec<f32>, // [layer][pos][kv_dim]
    v_cache: Vec<f32>,
    pub logits: Vec<f32>,
}

impl<'a> Session<'a> {
    pub fn new(m: &'a Model, max_seq: usize) -> Self {
        let c = &m.cfg;
        let (q_dim, kv_dim) = (c.n_heads * c.head_dim, c.n_kv_heads * c.head_dim);
        Session {
            m,
            pos: 0,
            max_seq,
            x: vec![0.0; c.dim],
            xb: vec![0.0; c.dim],
            xb2: vec![0.0; c.dim],
            qkv: vec![0.0; q_dim + 2 * kv_dim],
            attn_out: vec![0.0; q_dim],
            att: vec![0.0; c.n_heads * max_seq],
            hb: vec![0.0; 2 * c.hidden],
            rope: vec![(0.0, 0.0); c.head_dim / 2],
            k_cache: vec![0.0; c.n_layers * max_seq * kv_dim],
            v_cache: vec![0.0; c.n_layers * max_seq * kv_dim],
            logits: vec![0.0; c.vocab],
        }
    }

    /// Run one token through the model at position `self.pos`, appending to the KV cache.
    /// Skip `want_logits` for prompt tokens whose predictions you don't need (saves the vocab projection).
    pub fn step(&mut self, token: u32, want_logits: bool) {
        let m = self.m;
        let c = &m.cfg;
        let pos = self.pos;
        assert!(
            pos < self.max_seq,
            "context length {} exceeded",
            self.max_seq
        );
        assert!((token as usize) < c.vocab, "token {token} out of range");
        let (hd, q_dim, kv_dim, max_seq) = (
            c.head_dim,
            c.n_heads * c.head_dim,
            c.n_kv_heads * c.head_dim,
            self.max_seq,
        );
        let group = c.n_heads / c.n_kv_heads; // grouped-query attention: query heads per KV head
        let scale = 1.0 / (hd as f32).sqrt();

        self.x
            .copy_from_slice(&m.embed[token as usize * c.dim..][..c.dim]);
        for (i, r) in self.rope.iter_mut().enumerate() {
            let angle = pos as f64 * c.rope_theta.powf(-2.0 * i as f64 / hd as f64);
            *r = (angle.cos() as f32, angle.sin() as f32);
        }

        for (l, layer) in m.layers.iter().enumerate() {
            // --- attention ---
            rmsnorm(&mut self.xb, &self.x, &layer.attn_norm, c.eps);
            let layer_off = l * max_seq * kv_dim;
            let slot = layer_off + pos * kv_dim..layer_off + (pos + 1) * kv_dim;
            layer.wqkv.matvec(&self.xb, &mut self.qkv);
            let (q, kv) = self.qkv.split_at_mut(q_dim);
            let (k, v) = kv.split_at_mut(kv_dim);
            rope(q, hd, &self.rope);
            rope(k, hd, &self.rope);
            self.k_cache[slot.clone()].copy_from_slice(k);
            self.v_cache[slot].copy_from_slice(v);

            let (q, kc, vc) = (
                &self.qkv[..q_dim],
                &self.k_cache[layer_off..],
                &self.v_cache[layer_off..],
            );
            self.attn_out
                .par_chunks_mut(hd)
                .zip(self.att.par_chunks_mut(max_seq))
                .enumerate()
                .for_each(|(h, (out, att))| {
                    let qh = &q[h * hd..][..hd];
                    let kv = (h / group) * hd;
                    let att = &mut att[..=pos];
                    for (t, a) in att.iter_mut().enumerate() {
                        *a = dot(qh, &kc[t * kv_dim + kv..][..hd]) * scale;
                    }
                    softmax(att);
                    out.fill(0.0);
                    for (t, &a) in att.iter().enumerate() {
                        for (o, v) in out.iter_mut().zip(&vc[t * kv_dim + kv..][..hd]) {
                            *o += a * v;
                        }
                    }
                });
            layer.wo.matvec(&self.attn_out, &mut self.xb2);
            add(&mut self.x, &self.xb2);

            // --- SwiGLU MLP ---
            rmsnorm(&mut self.xb, &self.x, &layer.ffn_norm, c.eps);
            layer.w_gate_up.matvec(&self.xb, &mut self.hb);
            let (gate, up) = self.hb.split_at_mut(c.hidden);
            for (g, u) in gate.iter_mut().zip(&*up) {
                *g = *g / (1.0 + (-*g).exp()) * u;
            }
            layer.w_down.matvec(gate, &mut self.xb);
            add(&mut self.x, &self.xb);
        }
        self.pos += 1;

        if want_logits {
            rmsnorm(&mut self.xb, &self.x, &m.norm, c.eps);
            m.lm_head.matvec(&self.xb, &mut self.logits);
        }
    }
}

fn rmsnorm(out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    for ((o, v), w) in out.iter_mut().zip(x).zip(w) {
        *o = v * inv * w;
    }
}

/// Rotary embeddings, "rotate half" layout (what Hugging Face Llama checkpoints use):
/// dimension i is paired with i + head_dim/2.
fn rope(v: &mut [f32], hd: usize, table: &[(f32, f32)]) {
    let half = hd / 2;
    for head in v.chunks_exact_mut(hd) {
        let (lo, hi) = head.split_at_mut(half);
        for ((a, b), &(cos, sin)) in lo.iter_mut().zip(hi).zip(table) {
            (*a, *b) = (*a * cos - *b * sin, *a * sin + *b * cos);
        }
    }
}

pub fn softmax(x: &mut [f32]) {
    let max = x.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

fn add(x: &mut [f32], y: &[f32]) {
    for (a, b) in x.iter_mut().zip(y) {
        *a += b;
    }
}
