use rand::Rng;
// use ndarray::*;
// use ndarray_linalg::*;
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
pub extern "C" fn create_linear_model(mod_size: usize) -> *mut f32 {
    let mod_size_w = mod_size + 1;
    let mut model: Vec<f32> = Vec::with_capacity(mod_size_w);
    let mut rng = rand::thread_rng();

    for _i in 0..mod_size_w{
        let x = rng.gen_range(-1.0..1.0);
        model.push(x);
    }

    let boxed_model = model.into_boxed_slice();
    let model_ref = Box::leak(boxed_model);

    model_ref.as_mut_ptr()
}

#[no_mangle]
pub extern "C" fn predict_linear_model_regression(model: *mut f32, inputs: *mut f32, mod_size: usize) -> f32{
    let modelvec = unsafe{from_raw_parts(model, mod_size)};
    let inputvec = unsafe{from_raw_parts(inputs, mod_size)};


    let mut w_sum = modelvec[0];

    for i in 1..mod_size{
        w_sum += modelvec[i] * inputvec[(i-1)]
    }
    w_sum
}

#[no_mangle]
pub extern "C" fn predict_linear_model_classification(model: *mut f32, inputs: *mut f32, mod_size: usize) -> f32{
    let pred = predict_linear_model_regression(model, inputs, mod_size);
    let sign = if pred >= 0.0 {1.0} else {-1.0};

    sign
}


#[no_mangle]
pub extern "C" fn train_rosenblatt_linear_model(np_model: *mut f32, np_inputs: *mut f32, np_expected_outputs: *mut f32, mod_size: usize, dataset_size: usize, iterations: i32, alpha: f32) {
    let input_size = mod_size - 1;
    let sample_count = dataset_size / input_size;
    let mut rng = rand::thread_rng();

    let model= unsafe {from_raw_parts_mut(np_model, mod_size)};
    let inputs = unsafe {from_raw_parts_mut(np_inputs, dataset_size)};
    let expected_outputs = unsafe {from_raw_parts(np_expected_outputs, sample_count)};

    for _i in 0..iterations{
        let k = rng.gen_range(0..sample_count);

        let xk_slice = &mut inputs[(k*input_size)..((k+1)*input_size)];

        let xk = xk_slice.as_mut_ptr();

        let yk = expected_outputs[k];
        let gxk = predict_linear_model_classification(np_model, xk, mod_size);

        model[0 as usize] += alpha * (yk - gxk) * 1.0;

        for j in 1..mod_size{
            model[j] += alpha * (yk - gxk) * xk_slice[(j - 1) ];
        }
    }
}

#[no_mangle]
pub extern "C" fn destroy_model(model: *mut f32, mod_size: usize){
    unsafe{
        let _ = Vec::from_raw_parts(model, mod_size, mod_size);
    }
}

