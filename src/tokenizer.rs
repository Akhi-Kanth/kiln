//! Byte-level BPE tokenizer (GPT-2 style), read from a Hugging Face `tokenizer.json`.
//!
//! encode: split out special tokens -> pre-tokenize (digits, GPT-2 regex rules) ->
//!         map bytes to printable chars -> apply BPE merges by rank.

use serde_json::Value;
use std::{cmp::Reverse, collections::HashMap, fs, path::Path};

pub struct Tokenizer {
    vocab: HashMap<String, u32>,
    id_to_token: Vec<String>,
    /// (left, right) -> (merge rank, merged token id). Lower rank merges first.
    merges: HashMap<(u32, u32), (u32, u32)>,
    /// Added tokens like `<|im_start|>`, longest first so longer matches win ties.
    specials: Vec<(String, u32)>,
    /// Base token per byte; `None` if the vocab lacks it (HF silently drops those bytes too).
    byte_ids: [Option<u32>; 256],
    char_to_byte: HashMap<char, u8>,
}

impl Tokenizer {
    pub fn load(path: &Path) -> Result<Self, String> {
        let err = |e: String| format!("{}: {e}", path.display());
        let text = fs::read_to_string(path).map_err(|e| err(e.to_string()))?;
        let j: Value = serde_json::from_str(&text).map_err(|e| err(e.to_string()))?;
        let model = &j["model"];
        if model["type"] != "BPE" || j["pre_tokenizer"].to_string().contains("Metaspace") {
            return Err(err("only byte-level BPE tokenizers are supported".into()));
        }

        let mut vocab: HashMap<String, u32> = model["vocab"]
            .as_object()
            .ok_or_else(|| err("missing model.vocab".into()))?
            .iter()
            .filter_map(|(k, v)| Some((k.clone(), v.as_u64()? as u32)))
            .collect();
        let mut specials = Vec::new();
        for t in j["added_tokens"].as_array().into_iter().flatten() {
            if let (Some(s), Some(id)) = (t["content"].as_str(), t["id"].as_u64()) {
                specials.push((s.to_string(), id as u32));
                vocab.insert(s.to_string(), id as u32);
            }
        }
        specials.sort_by_key(|(s, _)| Reverse(s.len()));

        let mut id_to_token =
            vec![String::new(); vocab.values().max().map_or(0, |&m| m as usize + 1)];
        for (s, &id) in &vocab {
            id_to_token[id as usize] = s.clone();
        }

        let mut merges = HashMap::new();
        for (rank, m) in model["merges"].as_array().into_iter().flatten().enumerate() {
            // Older files store "a b", newer ones ["a", "b"].
            let pair = match m {
                Value::String(s) => s.split_once(' '),
                Value::Array(v) => v[0].as_str().zip(v[1].as_str()),
                _ => None,
            };
            let Some((a, b)) = pair else { continue };
            if let (Some(&l), Some(&r), Some(&out)) =
                (vocab.get(a), vocab.get(b), vocab.get(&format!("{a}{b}")))
            {
                merges.insert((l, r), (rank as u32, out));
            }
        }

        let byte_chars = bytes_to_unicode();
        let byte_ids = byte_chars.map(|c| vocab.get(&c.to_string()).copied());
        let char_to_byte = byte_chars
            .iter()
            .enumerate()
            .map(|(b, &c)| (c, b as u8))
            .collect();
        Ok(Tokenizer {
            vocab,
            id_to_token,
            merges,
            specials,
            byte_ids,
            char_to_byte,
        })
    }

    pub fn token_id(&self, s: &str) -> Option<u32> {
        self.vocab.get(s).copied()
    }

    /// Encode text. Special tokens written literally (e.g. `<|im_start|>`) become their ids.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            let next_special = self
                .specials
                .iter()
                .filter_map(|(s, id)| rest.find(s.as_str()).map(|i| (i, s.len(), *id)))
                .min_by_key(|&(i, len, _)| (i, Reverse(len)));
            let end = next_special.map_or(rest.len(), |(i, _, _)| i);
            for piece in pretokenize(&rest[..end]) {
                self.bpe(piece, &mut ids);
            }
            let Some((i, len, id)) = next_special else {
                break;
            };
            ids.push(id);
            rest = &rest[i + len..];
        }
        ids
    }

    fn bpe(&self, piece: &str, out: &mut Vec<u32>) {
        let mut syms: Vec<u32> = piece
            .bytes()
            .filter_map(|b| self.byte_ids[b as usize])
            .collect();
        // ponytail: O(n^2) per piece; pieces are words so n is tiny. Use a heap if that changes.
        while let Some((_, i, merged)) = syms
            .windows(2)
            .enumerate()
            .filter_map(|(i, w)| {
                self.merges
                    .get(&(w[0], w[1]))
                    .map(|&(rank, m)| (rank, i, m))
            })
            .min()
        {
            syms[i] = merged;
            syms.remove(i + 1);
        }
        out.extend(syms);
    }

    /// Raw bytes for one token. A multi-byte character can span tokens, so
    /// callers should buffer bytes until they form valid UTF-8.
    pub fn decode(&self, id: u32) -> Vec<u8> {
        let tok = self.id_to_token.get(id as usize).map_or("", String::as_str);
        let mut out = Vec::new();
        for c in tok.chars() {
            match self.char_to_byte.get(&c) {
                Some(&b) => out.push(b),
                None => out.extend(c.to_string().bytes()),
            }
        }
        out
    }
}

/// GPT-2's reversible byte -> printable char table, so BPE never sees raw whitespace/control bytes.
fn bytes_to_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut extra = 0;
    for b in 0..256u32 {
        let printable =
            (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        let c = if printable {
            b
        } else {
            extra += 1;
            255 + extra
        };
        table[b as usize] = char::from_u32(c).unwrap();
    }
    table
}

/// `Digits(individual_digits)` then GPT-2's pattern:
/// `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
/// Hand-rolled because Rust's regex crate has no lookahead.
fn pretokenize(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    for (i, c) in text.char_indices() {
        if c.is_numeric() {
            gpt2_split(&text[start..i], &mut out);
            out.push(&text[i..i + c.len_utf8()]);
            start = i + c.len_utf8();
        }
    }
    gpt2_split(&text[start..], &mut out);
    out
}

fn gpt2_split<'a>(s: &'a str, out: &mut Vec<&'a str>) {
    // ponytail: char::is_alphabetic is Unicode "Alphabetic", slightly wider than \p{L}
    // (it includes some combining marks). Exact for ASCII and nearly all real text.
    let letter = |c: char| c.is_alphabetic();
    let number = |c: char| c.is_numeric();
    let other = |c: char| !c.is_whitespace() && !c.is_alphabetic() && !c.is_numeric();
    let cs: Vec<(usize, char)> = s.char_indices().collect();
    let n = cs.len();
    let byte = |i: usize| if i < n { cs[i].0 } else { s.len() };
    let run = |from: usize, f: &dyn Fn(char) -> bool| (from..n).find(|&j| !f(cs[j].1)).unwrap_or(n);

    let mut i = 0;
    while i < n {
        let c = cs[i].1;
        if c == '\'' {
            let rest = &s[cs[i].0..];
            if let Some(p) = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"]
                .iter()
                .find(|p| rest.starts_with(**p))
            {
                out.push(&rest[..p.len()]);
                i += p.len();
                continue;
            }
        }
        // An optional single leading space attaches to the following word/number/punctuation.
        let k = if c == ' ' && i + 1 < n && !cs[i + 1].1.is_whitespace() {
            i + 1
        } else {
            i
        };
        let ck = cs[k].1;
        let end = if letter(ck) {
            run(k, &letter)
        } else if number(ck) {
            run(k, &number)
        } else if other(ck) {
            run(k, &other)
        } else {
            // Whitespace run. `\s+(?!\S)` leaves the last space for the next word.
            let e = run(i, &char::is_whitespace);
            if e == n || e - i == 1 {
                e
            } else {
                e - 1
            }
        };
        out.push(&s[byte(i)..byte(end)]);
        i = end;
    }
}
