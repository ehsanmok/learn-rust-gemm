use std::ops::Add;
use blas::ddot;
use cblas::ddot as cddot;
use ndarray::prelude::*;
use ndarray_rand::{RandomExt, F32};
use rand::distributions::Normal;
use rayon::prelude::*;

pub fn aslice_dot_naive(a: &[f64], b: &[f64]) -> f64 {
    let mut ret = 0.0;
    for i in 0..a.len() {
        ret += a[i] * b[i];
    }
    ret
}

pub fn vec_dot_naive(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    let mut ret = 0.0;
    for i in 0..a.len() {
        ret += a[i] * b[i];
    }
    ret
}

pub fn vec_dot_zip(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

pub fn vec_dot_fold(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    (0..a.len()).fold(0f64, |sum, i| sum + a[i] * b[i])
}

pub fn vec_dot_unsafe(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    unsafe {
        (0..a.len()).fold(0f64, |sum, i| sum + a.get_unchecked(i) * b.get_unchecked(i))
    }
}

pub fn vec_dot_par_iter_slow(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    a.par_iter().zip(b.par_iter()).map(|(&x, &y)| x * y).sum()
}

pub fn vec_dot_par_iter_fast(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    a.par_iter().zip(b.par_iter()).map(|(&x, &y)| x * y).reduce_with(Add::add).unwrap()
}

pub fn dot_blas(a: &[f64], b: &[f64]) -> f64 {
    unsafe {
        ddot(a.len() as i32, &a[..], 1, &b[..], 1)
    }
}

pub fn dot_cblas(v: &Vec<f64>, w: &Vec<f64>) -> f64 {
    unsafe {
        cddot(v.len() as i32, &v[..], 1, &w[..], 1)
    }
}


#[cfg(test)]
mod tests {

    use std::iter;
    use rand::{Rng, thread_rng};
    use super::*;

    fn randn_vec(n: usize) -> Vec<f64> {
        let mut rng = thread_rng();
        iter::repeat(()).map(|()| rng.gen()).take(n).collect::<Vec<f64>>()
    }

    macro_rules! all_close {
        ($x:expr, $y:expr) => (assert!(($x - $y).abs() < 1e-8));
        ($x:expr, $y:expr, $($ys:expr),+) => (
            all_close!($y, $($ys),+)
        )
    }

    #[test]
    fn dot() {
        let v = randn_vec(10);
        let a =  aslice_dot_naive(&v[..], &v[..]);
        let b = vec_dot_naive(&v, &v);
        let c = vec_dot_zip(&v, &v);
        let d = vec_dot_fold(&v, &v);
        let e = vec_dot_unsafe(&v, &v);
        let f = vec_dot_par_iter_slow(&v, &v);
        let g = vec_dot_par_iter_fast(&v, &v);
        let h = dot_blas(&v, &v);
        let i = dot_cblas(&v, &v);
        all_close!(a, b, c, d, e, f, g, h, i)
    }

    // #[bench] // needs nightly test feature
    // fn bench_aslice_dot_naive(bench: &mut test::Bencher) {
    //     let n = 1000000;
    //     let v = randn_vec(n);
    //     bench.iter(|| {
    //         aslice_dot_naive(&v[..], &v[..])
    //     });
    // }
    //
}
