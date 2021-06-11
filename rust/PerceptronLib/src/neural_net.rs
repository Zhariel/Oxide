use rand::Rng;
use std::slice::{from_raw_parts, from_raw_parts_mut};

pub struct NeuralNet{
    layer_count : usize,
    input_count : usize,
    neuron_count : usize,
    x: Vec<Vec<f32>>,
    w: Vec<Vec<Vec<f32>>>,
    deltas : Vec<f32>
}

#[no_mangle]
pub extern "C" fn create_NeuralNet(input_count: usize, neuron_count: usize, layer_count : usize) -> *mut NeuralNet{
    let mut rng = rand::thread_rng();

    let mut pmc = NeuralNet{
        layer_count,
        input_count,
        neuron_count,
        // x: vec![vec![0.0; max_neur_count]; layer_count],
        // w: vec![vec![vec![0.0; max_neur_count];max_neur_count]; layer_count],
        x: Vec::with_capacity(layer_count),
        w: Vec::with_capacity(layer_count),
        deltas: Vec::new()
    };

    //Init X & W
    for l in 0..layer_count{
        let size = if l == 0 {input_count} else {neuron_count};
        pmc.x[l] = Vec::with_capacity(input_count);
        pmc.w[l] = Vec::with_capacity(input_count);

        for i in 0..size{
            pmc.w[l][i] = Vec::with_capacity(neuron_count);
            for j in 0..neuron_count{
                pmc.w[l][i][j] = rng.gen_range(0.0..1.0) * 2.0 - 1.0;
            }
        }

        pmc.x[l][0] = 1.0;
    }

    let boxed_pmc = Box::new(pmc);
    let ref_to_pmc = Box::leak(boxed_pmc);

    ref_to_pmc
}

#[no_mangle]
pub extern "C" fn train_NeuralNet(pmc_raw: *mut NeuralNet, x: *mut f32, x_length: usize, y: *mut f32, y_length: usize, sample_count: usize, epoch: usize, alpha: f32){
    let pmc = unsafe {
        pmc_raw.as_mut().unwrap()
    };
    let mut rng = rand::thread_rng();
    let inputs = unsafe {from_raw_parts_mut(x, x_length)};
    let expected = unsafe {from_raw_parts_mut(y, y_length)};

    for e in 0..epoch{
        for i in 0..sample_count{
            for d in 0..pmc.input_count{
                pmc.x[0][d] = inputs[e + d];
            }

            feed_Forward(pmc);


        }
    }
}

#[no_mangle]
pub extern "C" fn mse(given: f32, desired: f32) -> f32{
    let diff = given - desired;
    diff * diff
}

#[no_mangle]
pub extern "C" fn backprop_Classification(){

}

#[no_mangle]
pub extern "C" fn feed_Forward(pmc_raw: *mut NeuralNet){
    let pmc = unsafe {
        pmc_raw.as_mut().unwrap()
    };

    for l in 1..pmc.layer_count - 1{
        for i in 0..pmc.neuron_count {
            let mut sum: f32 = 0.0;

            for j in 0..pmc.neuron_count {
                sum += pmc.w[l][i][j] * pmc.x[l][j];
            }

            pmc.x[l + 1][i] = sum.tanh();
        }
    }
}

#[no_mangle]
pub extern "C" fn release_NeuralNet(pmc: *mut NeuralNet){
    unsafe {
        Box::from_raw(pmc);
    }
}