use rand::Rng;
use ndarray::*;
use ndarray_linalg::*;
use std::slice::{from_raw_parts, from_raw_parts_mut};

#[cfg(test)]
mod tests {
    // #[test]
    // fn it_works() {
    //     assert_eq!(2 + 2, 4);
    // }
}

#[no_mangle]
pub extern "C" fn sum(arr: *mut i32, size: i32) -> i32 {
    let arr = unsafe {from_raw_parts(arr, size as usize)};
    arr.iter().fold(0, |acc, elt| acc + elt)
}

#[no_mangle]
pub extern "C" fn create_linear_model(mod_size: i32) -> *mut f64 {
    let mod_size_w = mod_size + 1;
    let mut model: Vec<f64> = Vec::with_capacity(mod_size_w as usize);
    let mut rng = rand::thread_rng();

    for _i in 0..mod_size_w{
        let x = rng.gen_range(0.0..1.0) * 2.0 - 1.0;
        model.push(x);
    }

    let boxed_model = model.into_boxed_slice();
    let model_ref = Box::leak(boxed_model);

    model_ref.as_mut_ptr()
}

#[no_mangle]
pub extern "C" fn predict_linear_model_regression(model: *mut f64, inputs: *mut f64, mod_size: i32) -> f64{
    let modelvec = unsafe{from_raw_parts(model, mod_size as usize)};
    let inputvec = unsafe{from_raw_parts(inputs, mod_size as usize)};

    // modelvec.iter().zip(inputvec.into_iter()).map(|(x, y)| x * y).fold(0.0, |sum, i| sum + i)

    let mut w_sum = modelvec[0];

    for i in 1..mod_size{
        w_sum += modelvec[i as usize] * inputvec[(i-1) as usize]
    }
    w_sum
}

#[no_mangle]
pub extern "C" fn predict_linear_model_classification(model: *mut f64, inputs: *mut f64, mod_size: i32) -> f64{
    let pred = predict_linear_model_regression(model, inputs, mod_size);
    let sign = if pred >= 0.0 {1.0} else {-1.0};

    sign
}

#[no_mangle]
pub extern "C" fn train_rosenblatt_linear_model(np_model: *mut f64, np_inputs: *mut f64, np_expected_outputs: *mut f64, mod_size: usize, dataset_size: usize, iterations: i32, alpha: f64) -> *mut f64 {
    let model= unsafe {from_raw_parts_mut(np_model, mod_size)};
    let inputs = unsafe {from_raw_parts_mut(np_inputs, dataset_size)};
    let expected_outputs = unsafe {from_raw_parts(np_expected_outputs, mod_size)};

    let input_size = mod_size - 1;
    let sample_count = dataset_size / input_size;
    let mut rng = rand::thread_rng();

    for _i in 0..iterations{
        let k = rng.gen_range(0..sample_count);

        let xk_start = (k*input_size) as usize;
        let xk_end = (k*input_size+input_size) as usize;
        let xk_slice = &mut inputs[xk_start..xk_end];

        let xk = xk_slice.as_mut_ptr();

        let yk = expected_outputs[k as usize];
        let gxk = predict_linear_model_classification(np_model, xk, mod_size as i32);

        model[0 as usize] += alpha * (yk - gxk) * 1.0;

        for j in 1..mod_size{
            model[j] += alpha * (yk - gxk) * inputs[(xk_start + j - 1) ];
        }
    }

    // destroy_model(np_model, mod_size);

    // let mut new_model: Vec<f64> = Vec::with_capacity(mod_size as usize);
    // for i in 0..mod_size{
    //     new_model[i as usize] = model[i as usize];
    // }
    // let boxed_model = new_model.into_boxed_slice();
    // let model_ref = Box::leak(boxed_model);

    model.as_mut_ptr()
}

#[no_mangle]
pub extern "C" fn train_regression_linear_model(model: *mut f64, inputs: *mut f64, expected_outputs: *mut f64, mod_size: i32, dataset_size: i32) {
    let mod_size_w = mod_size + 1;
    let input_size = mod_size;
}

#[no_mangle]
pub extern "C" fn destroy_model(model: *mut f64, mod_size: i32){
    unsafe{
        let _ = Vec::from_raw_parts(model, mod_size as usize, mod_size as usize);
    }
}

