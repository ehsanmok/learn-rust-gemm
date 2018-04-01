use criterion::Criterion;
use criterion::Fun;
use std::iter;
use learn_rust_gemm::vector::*;
use rand::{Rng, thread_rng};


fn randn_vec(n: usize) -> Vec<f64> {
    let mut rng = thread_rng();
    iter::repeat(()).map(|()| rng.gen()).take(n).collect::<Vec<f64>>()
}

fn compare_dots(c: &mut Criterion) {
    let v = randn_vec(1_000);

    let aslice_naive = Fun::new("Slice naive",
                        |b, v: &Vec<f64>| b.iter(|| aslice_dot_naive(&v[..], &v[..])));
    let vec_naive = Fun::new("Vec naive",
                                |b, v: &Vec<f64>| b.iter(|| vec_dot_naive(v, v)));
    let vec_naive_zip = Fun::new("Vec naive zip",
                                    |b, v: &Vec<f64>| b.iter(|| vec_dot_zip(v, v)));
    let vec_naive_fold = Fun::new("Vec naive fold",
                                    |b, v: &Vec<f64>| b.iter(|| vec_dot_fold(&v, &v)));
    let vec_naive_unsafe = Fun::new("Vec naive fold unsafe" ,
                                    |b, v: &Vec<f64>| b.iter(|| vec_dot_unsafe(&v, &v)));
    let vec_naive_par_iter_slow = Fun::new("Vec naive parallel slow",
                                        |b, v: &Vec<f64>| b.iter(|| vec_dot_par_iter_slow(v, v)));
    let vec_naive_par_iter_fast = Fun::new("Vec naive parallel fast",
                                    |b, v: &Vec<f64>| b.iter(|| vec_dot_par_iter_fast(v, v)));
    let vec_blas = Fun::new("Blas", |b, v: &Vec<f64>| b.iter(|| dot_blas(&v, &v)));
    let vec_cblas = Fun::new("CBlas", |b, v: &Vec<f64>| b.iter(|| dot_cblas(&v, &v)));

    let dots = vec![aslice_naive, vec_naive,
                    vec_naive_zip, vec_naive_fold, vec_naive_unsafe,
                    vec_naive_par_iter_slow, vec_naive_par_iter_fast,
                    vec_blas, vec_cblas];
    c.bench_functions("Dots", dots, v);
}

criterion_group!(benches, compare_dots);
