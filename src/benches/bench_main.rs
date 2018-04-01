#[macro_use]
extern crate criterion;
extern crate walkdir;
extern crate ndarray;
extern crate ndarray_rand;
extern crate learn_rust_gemm;
extern crate rand;

mod plain;
mod ndarray_dot;

criterion_main!{
    plain::benches,
    ndarray_dot::bench
}
