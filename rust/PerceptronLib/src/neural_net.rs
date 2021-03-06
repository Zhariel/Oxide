use rand::Rng;
use std::slice::{from_raw_parts, from_raw_parts_mut};
use serde::{Serialize, Deserialize};
use std::fs;
use std::fs::File;
use std::io::Read;

#[derive(Serialize, Deserialize)]
pub struct NeuralNet{
    layer_count : usize,
    input_len : usize,
    hidden_len : usize,
    output_len : usize,
    x: Vec<Vec<f32>>,
    w: Vec<Vec<Vec<f32>>>,
    out: Vec<f32>,
    deltas : Vec<Vec<f32>>
}

fn init_x(layer_count: usize, input_len: usize, hidden_len : usize) -> Vec<Vec<f32>>{
    let mut x:Vec<Vec<f32>> = vec![];

    let input_with_bias = input_len + 1;
    let hidden_with_bias = hidden_len + 1;

    let layer: Vec<f32> = vec![0.0; input_with_bias];
    x.push(layer);
    x[0][0] = 1.0;

    for l in 1..layer_count{
        let layer: Vec<f32> = vec![0.0; hidden_with_bias];
        x.push(layer);
        x[l][0] = 1.0;
    }

    x
}

fn init_w(layer_count: usize, input_len: usize, hidden_len: usize, output_len: usize) -> Vec<Vec<Vec<f32>>>{
    let mut w:Vec<Vec<Vec<f32>>> = vec![];
    let input_with_bias = input_len + 1;
    let hidden_with_bias = hidden_len + 1;

    for l in 0..layer_count{
        // i prend la valeur de la longueur de la couche x suivante
        let size_i = if l == layer_count - 1 {output_len} else {hidden_with_bias};
        let layer : Vec<Vec<f32>> = vec![];
        w.push(layer);

        for i in 0..size_i{
            // j prend la valeur de la longueur de la couche x précédente
            let size_j = if l == 0 {input_with_bias} else {hidden_with_bias};
            let layer_i : Vec<f32> = vec![0.0; size_j];
            w[l].push(layer_i);

            //init j
            randomize_weights(&mut w[l][i]);
        }
    }
    w
}

fn init_deltas(layer_count: usize, hidden_len: usize, output_len: usize) -> Vec<Vec<f32>> {
    let mut deltas : Vec<Vec<f32>> = vec![];
    let hidden_with_bias = hidden_len + 1;

    for l in 0..layer_count{
        let size_i = if l == layer_count - 1 {output_len} else {hidden_with_bias};
        let layer: Vec<f32> = vec![0.0; size_i];
        deltas.push(layer);
    }

    deltas
}

fn randomize_weights(vec: &mut Vec<f32>){
    let mut rng = rand::thread_rng();
    for i in 0..vec.len(){
        vec[i] = rng.gen_range(-1.0..1.0);
    }
}

#[no_mangle]
pub extern "C" fn create_NeuralNet(layer_count: usize, input_len: usize, hidden_len : usize, output_len : usize) -> *mut NeuralNet{

    let nn = NeuralNet{
        layer_count,
        input_len,
        hidden_len,
        output_len,
        x: init_x(layer_count, input_len, hidden_len),
        w: init_w(layer_count, input_len, hidden_len, output_len),
        out: vec![0.0; output_len],
        deltas: init_deltas(layer_count, hidden_len, output_len)
    };

    let boxed_nn = Box::new(nn);
    let ref_to_nn = Box::leak(boxed_nn);

    ref_to_nn
}

#[no_mangle]
pub extern "C" fn store_NeuralNet(nn: &mut NeuralNet, f: i32){
    let serialized = serde_json::to_string(&nn).unwrap();
    let filename = "neural\\neural_".to_owned() + &f.to_string() + ".json";

    fs::write(filename, serialized).expect("Unable to write");
}

#[no_mangle]
pub extern "C" fn load_NeuralNet(f: i32) -> *mut NeuralNet{
    let filename = "neural\\neural_".to_owned() + &f.to_string() + ".json";

    let mut data = String::new();
    let mut file = File::open(filename).unwrap();
    file.read_to_string(&mut data).unwrap();

    let nn: NeuralNet = serde_json::from_str(&data).expect("Unable to read");

    let boxed_nn = Box::new(nn);
    let ref_to_nn = Box::leak(boxed_nn);

    ref_to_nn
}

#[no_mangle]
pub extern "C" fn train_NeuralNet(nn_raw: *mut NeuralNet, x: *mut f32, x_length: usize, y: *mut f32, y_length: usize, epoch: usize, alpha: f32, is_classif: bool){
    let nn = unsafe {
        nn_raw.as_mut().unwrap()
    };
    let inputs = unsafe {from_raw_parts_mut(x, x_length)};
    let expected = unsafe {from_raw_parts_mut(y, y_length)};
    let sample_count = x_length / nn.input_len;

    for e in 0..epoch{
        for i in 0..sample_count{
            //fit inputs in the first row of x
            for d in 1..nn.input_len+1{
                nn.x[0][d] = inputs[(i * nn.input_len) + d-1];
            }

            feed_forward(nn, is_classif);

            let mut y_sample: Vec<f32> = vec![0.0; nn.output_len];
            fill_vec(&mut expected[i*nn.output_len], &mut y_sample, nn.output_len);

            backprop(nn, &y_sample, alpha, is_classif);
        }
    }
}

#[no_mangle]
pub extern "C" fn predict(nn: &mut NeuralNet, x: *mut f32, is_classif: bool) -> *mut f32 {
    let inputs = unsafe {from_raw_parts_mut(x, nn.input_len)};

    //fit inputs in the first row of x
    for d in 1..nn.input_len+1{
        nn.x[0][d] = inputs[d-1];
    }

    feed_forward(nn, is_classif);

    let result = nn.out.clone();

    let boxed_result = result.into_boxed_slice();
    let result_ref = Box::leak(boxed_result);

    result_ref.as_mut_ptr()
}

fn feed_forward(nn: &mut NeuralNet, is_classif: bool){

    for l in 0..nn.layer_count-1{
        for i in 1..nn.w[l].len() {
            let mut w_sum: f32 = 0.0;

            for j in 0..nn.w[l][i].len() {
                w_sum += nn.w[l][i][j] * nn.x[l][j];
            }

            // let mut w_sum: f32 = nn.w[l][i].par_iter_mut()
            //     .zip(nn.x[l].par_iter_mut())
            //     .map(|(w, x)| *w * *x)
            //     .fold(|| 0.0, |sum, i| sum + i)
            //     .sum::<f32>();

            nn.x[l + 1][i] = w_sum.tanh();
        }
    }

    for o in 0..nn.output_len{
        let mut output = 0.0;
        let cur_l_idx = nn.layer_count-1;
        let cur_l_len = nn.x[cur_l_idx].len();
        for j in 0..cur_l_len{
            output += nn.w[cur_l_idx][o][j] * nn.x[cur_l_idx][j];
        }

        if is_classif{
            nn.out[o] = output.tanh();
        }
        else {
            nn.out[o] = output;
        }
    }
}

fn backprop(nn: &mut NeuralNet, y: &Vec<f32>, alpha: f32, is_classif: bool){
    let mut idx = nn.layer_count - 1;

    if is_classif{
        deltas_last_layer_classif(nn, &y);
    }
    else {
        deltas_last_layer_regression(nn, &y);
    }

    for l in 1..nn.layer_count{
        idx = nn.layer_count - l - 1;

        deltas_hidden_layer(nn, idx);
    }

    correct_w(nn, alpha);
}

fn correct_w(nn: &mut NeuralNet, alpha: f32){
    for l in 0..nn.layer_count{
        for i in 0..nn.w[l].len(){
            for j in 0..nn.w[l][i].len(){
                nn.w[l][i][j] = nn.w[l][i][j] - alpha * nn.x[l][j] * nn.deltas[l][i];
            }
        }
    }
}

fn deltas_hidden_layer(nn: &mut NeuralNet, idx: usize){
    for i in 0..nn.deltas[idx].len(){
        let mut sum_deltas = 0.0;

        for j in 0..nn.deltas[idx+1].len(){
            sum_deltas += nn.w[idx+1][j][i] * nn.deltas[idx+1][j];
        }

        nn.deltas[idx][i] = (1.0 - nn.x[idx+1][i].powf(2.0)) * sum_deltas;
    }
}

fn deltas_last_layer_classif(nn: &mut NeuralNet, y: &Vec<f32>){
    let idx = nn.layer_count - 1;

    for i in 0..nn.deltas[idx].len(){
        nn.deltas[idx][i] = (1.0 - nn.out[i].powf(2.0)) * (nn.out[i] - y[i]);
    }
}

fn deltas_last_layer_regression(nn: &mut NeuralNet, y: &Vec<f32>){
    let idx = nn.layer_count - 1;

    for i in 0..nn.deltas[idx].len(){
        nn.deltas[idx][i] = nn.out[i] - y[i];
    }
}

fn fill_vec(origin_ptr: *mut f32, y: &mut Vec<f32>, len: usize){
    let origin = unsafe {from_raw_parts_mut(origin_ptr, len)};
    for i in 0..len{
        y[i] = origin[i];
    }
}

#[no_mangle]
pub extern "C" fn release_NeuralNet(nn: *mut NeuralNet){
    unsafe {
        Box::from_raw(nn);
    }
}

#[no_mangle]
pub extern "C" fn release_result(result: *mut f32, size: usize){
    unsafe{
        let _ = Vec::from_raw_parts(result, size, size);
    }
}