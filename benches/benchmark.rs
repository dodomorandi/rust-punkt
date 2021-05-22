#[cfg(not(feature = "bench"))]
compile_error! {"Please add `--features bench` to your command in order to run benches"}

extern crate criterion;
extern crate punkt;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use punkt::{
    prelude::Standard, train_on_document, SentenceTokenizer, Token, Trainer, TrainingData,
    WordTokenizer,
};

fn bench_word_tokenizer(s: &str) -> Vec<Token> {
    let t: WordTokenizer<Standard> = WordTokenizer::new(s);
    t.collect()
}

fn bench_sentence_tokenizer<'a>(doc: &'a str, data: &'a mut TrainingData) -> Vec<&'a str> {
    train_on_document(data, doc);

    let iter: SentenceTokenizer<Standard> = SentenceTokenizer::new(doc, data);
    iter.collect()
}

fn bench_trainer(doc: &str) {
    let mut data = TrainingData::new();
    let trainer: Trainer<Standard> = Trainer::new();

    trainer.train(doc, &mut data);
}

fn word_tokenizer(c: &mut Criterion) {
    c.bench_function("word tokenizer short", |b| {
        b.iter(|| bench_word_tokenizer(black_box(include_str!("../test/raw/sigma-wiki.txt"))))
    });

    c.bench_function("word tokenizer medium", |b| {
        b.iter(|| bench_word_tokenizer(black_box(include_str!("../test/raw/npr-article-01.txt"))))
    });

    c.bench_function("word tokenizer long", |b| {
        b.iter(|| {
            bench_word_tokenizer(black_box(include_str!(
                "../test/raw/the-sayings-of-confucius.txt"
            )))
        })
    });

    c.bench_function("word tokenizer very long", |b| {
        b.iter(|| {
            bench_word_tokenizer(black_box(include_str!(
                "../test/raw/pride-and-prejudice.txt"
            )))
        })
    });
}

fn sentence_tokenizer(c: &mut Criterion) {
    c.bench_function("train on document short", |b| {
        b.iter(|| {
            let mut data = TrainingData::new();
            bench_sentence_tokenizer(
                black_box(include_str!("../test/raw/sigma-wiki.txt")),
                &mut data,
            );
        })
    });

    c.bench_function("train on document medium", |b| {
        b.iter(|| {
            let mut data = TrainingData::new();
            bench_sentence_tokenizer(
                black_box(include_str!("../test/raw/npr-article-01.txt")),
                &mut data,
            );
        })
    });

    c.bench_function("train on document long", |b| {
        b.iter(|| {
            let mut data = TrainingData::new();
            bench_sentence_tokenizer(
                black_box(include_str!("../test/raw/pride-and-prejudice.txt")),
                &mut data,
            );
        })
    });
}

fn trainer(c: &mut Criterion) {
    c.bench_function("trainer short", |b| {
        b.iter(|| bench_trainer(black_box(include_str!("../test/raw/sigma-wiki.txt"))))
    });

    c.bench_function("word tokenizer medium", |b| {
        b.iter(|| bench_trainer(black_box(include_str!("../test/raw/npr-article-01.txt"))))
    });

    c.bench_function("word tokenizer long", |b| {
        b.iter(|| {
            bench_trainer(black_box(include_str!(
                "../test/raw/the-sayings-of-confucius.txt"
            )))
        })
    });

    c.bench_function("word tokenizer very long", |b| {
        b.iter(|| {
            bench_trainer(black_box(include_str!(
                "../test/raw/pride-and-prejudice.txt"
            )))
        })
    });
}

criterion_group!(benches, word_tokenizer, sentence_tokenizer, trainer);
criterion_main!(benches);
