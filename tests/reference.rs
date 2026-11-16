//! Checks kiln against Hugging Face (tokenizers + PyTorch) outputs in tests/reference.json.
//! Regenerate with scripts/reference.py. Skipped if the model isn't downloaded.

use kiln::{model::Model, model::Session, tokenizer::Tokenizer};
use serde_json::Value;
use std::path::PathBuf;

fn model_dir() -> Option<PathBuf> {
    let dir =
        PathBuf::from(std::env::var("KILN_MODEL").unwrap_or_else(|_| "models/smollm2-135m".into()));
    dir.join("config.json").exists().then_some(dir)
}

#[test]
fn matches_huggingface() {
    let Some(dir) = model_dir() else {
        eprintln!("skipping: run scripts/download.sh first");
        return;
    };
    let reference: Value = serde_json::from_str(include_str!("reference.json")).unwrap();
    let tok = Tokenizer::load(&dir.join("tokenizer.json")).unwrap();

    for case in reference["tokenize"].as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        let want: Vec<u32> = case["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(tok.encode(text), want, "tokenizing {text:?}");
    }

    for q8 in [false, true] {
        let model = Model::load(&dir, q8).unwrap();
        for case in reference["logits"].as_array().unwrap() {
            let ids = tok.encode(case["prompt"].as_str().unwrap());
            let mut s = Session::new(&model, 512);
            for (i, &id) in ids.iter().enumerate() {
                s.step(id, i + 1 == ids.len());
            }
            let top: Vec<(usize, f32)> = case["top5"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| {
                    (
                        p[0].as_u64().unwrap() as usize,
                        p[1].as_f64().unwrap() as f32,
                    )
                })
                .collect();
            let argmax = s
                .logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0;
            assert_eq!(argmax, top[0].0, "q8={q8}: top token differs from PyTorch");
            let tol = if q8 { 0.5 } else { 1e-2 };
            for (id, want) in top {
                let got = s.logits[id];
                assert!(
                    (got - want).abs() < tol,
                    "q8={q8}: logit[{id}] = {got}, PyTorch says {want}"
                );
            }
        }
    }
}
