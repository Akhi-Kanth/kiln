use kiln::{model::Model, model::Session, tokenizer::Tokenizer};
use std::{io::Write, path::Path, str::FromStr, time::Instant};

const USAGE: &str = "usage: kiln <model_dir> [-p PROMPT] [--chat] [-n MAX_NEW_TOKENS] [--temp T] [--top-p P] [--seed S] [--ctx N] [--q8]";

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}\n{USAGE}");
        std::process::exit(1);
    }
}

fn num<T: FromStr>(s: String) -> Result<T, String> {
    s.parse().map_err(|_| format!("not a number: {s}"))
}

fn run() -> Result<(), String> {
    let (mut dir, mut prompt, mut chat, mut q8) =
        (None, "Once upon a time".to_string(), false, false);
    let (mut max_new, mut ctx, mut temp, mut top_p) = (256usize, 2048usize, 0.7f32, 0.9f32);
    let mut seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().ok_or_else(|| format!("{a} needs a value"));
        match a.as_str() {
            "-p" | "--prompt" => prompt = val()?,
            "-n" => max_new = num(val()?)?,
            "--ctx" => ctx = num(val()?)?,
            "--temp" => temp = num(val()?)?,
            "--top-p" => top_p = num(val()?)?,
            "--seed" => seed = num(val()?)?,
            "--chat" => chat = true,
            "--q8" => q8 = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            _ if dir.is_none() && !a.starts_with('-') => dir = Some(a),
            _ => return Err(format!("unknown argument {a}")),
        }
    }
    let dir = dir.ok_or("missing model_dir")?;
    let dir = Path::new(&dir);

    let t = Instant::now();
    let model = Model::load(dir, q8)?;
    let tok = Tokenizer::load(&dir.join("tokenizer.json"))?;
    let c = &model.cfg;
    eprintln!(
        "loaded {} ({} layers, dim {}, {}) in {:.2}s",
        dir.display(),
        c.n_layers,
        c.dim,
        if model.is_quantized() { "q8" } else { "f32" },
        t.elapsed().as_secs_f64()
    );

    let text = if chat {
        format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
    } else {
        prompt.clone()
    };
    let ids = tok.encode(&text);
    if ids.is_empty() || ids.len() >= ctx {
        return Err(format!("prompt is {} tokens; need 1..{ctx}", ids.len()));
    }
    let mut stop = c.eos.clone();
    stop.extend(tok.token_id("<|endoftext|>"));

    let mut s = Session::new(&model, ctx);
    let mut sampler = Sampler {
        temp,
        top_p,
        rng: seed | 1,
    };
    let mut stdout = std::io::stdout().lock();
    if !chat {
        print!("{prompt}");
    }

    let t = Instant::now();
    for (i, &id) in ids.iter().enumerate() {
        s.step(id, i + 1 == ids.len());
    }
    let prefill = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let (mut steps, mut pending) = (0, Vec::new());
    for generated in 1..=max_new {
        let next = sampler.sample(&s.logits);
        if stop.contains(&next) {
            break;
        }
        // Tokens can split a UTF-8 character; print only complete characters.
        pending.extend(tok.decode(next));
        let valid = match std::str::from_utf8(&pending) {
            Ok(_) => pending.len(),
            Err(e) => e.valid_up_to(),
        };
        stdout
            .write_all(&pending[..valid])
            .and_then(|_| stdout.flush())
            .map_err(|e| e.to_string())?;
        pending.drain(..valid);
        if generated == max_new || s.pos == ctx {
            break;
        }
        s.step(next, true);
        steps += 1;
    }
    let decode = t.elapsed().as_secs_f64();
    println!();
    eprintln!(
        "prefill: {} tok in {:.2}s ({:.1} tok/s) | decode: {} tok in {:.2}s ({:.1} tok/s)",
        ids.len(),
        prefill,
        ids.len() as f64 / prefill,
        steps,
        decode,
        steps as f64 / decode.max(1e-9)
    );
    Ok(())
}

struct Sampler {
    temp: f32,
    top_p: f32,
    rng: u64,
}

impl Sampler {
    /// xorshift64* -> uniform f32 in [0, 1).
    fn uniform(&mut self) -> f32 {
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        (self.rng.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Greedy when temp <= 0, otherwise temperature + nucleus (top-p) sampling.
    fn sample(&mut self, logits: &[f32]) -> u32 {
        let (argmax, &max) = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap();
        if self.temp <= 0.0 {
            return argmax as u32;
        }
        let probs: Vec<f32> = logits
            .iter()
            .map(|&l| ((l - max) / self.temp).exp())
            .collect();
        let total: f32 = probs.iter().sum();
        // Tokens below this can never be inside the nucleus; dropping them avoids sorting the whole vocab.
        let cutoff = (1.0 - self.top_p) / (probs.len() - 1) as f32 * total;
        let mut cand: Vec<(f32, u32)> = probs
            .iter()
            .enumerate()
            .filter(|(_, &p)| p >= cutoff)
            .map(|(i, &p)| (p, i as u32))
            .collect();
        cand.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        let (mut cum, mut keep) = (0.0, cand.len());
        for (i, (p, _)) in cand.iter().enumerate() {
            cum += p;
            if cum >= self.top_p * total {
                keep = i + 1;
                break;
            }
        }
        let mut r = self.uniform() * cand[..keep].iter().map(|c| c.0).sum::<f32>();
        for &(p, id) in &cand[..keep] {
            r -= p;
            if r <= 0.0 {
                return id;
            }
        }
        cand[keep - 1].1
    }
}
