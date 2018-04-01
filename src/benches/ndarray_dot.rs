use criterion::Criterion;
use ndarray::prelude::*;
use ndarray_rand::{RandomExt, F32};
use rand::distributions::Normal;


fn ndarray_dot_bench(c: &mut Criterion) {
    let w = Array::random(1_000, F32(Normal::new(0., 1.)));
    c.bench_function("Ndarray", move |b| b.iter(|| w.dot(&w)));
}

criterion_group!(bench, ndarray_dot_bench);
