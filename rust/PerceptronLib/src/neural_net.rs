use rand::Rng;
use std::slice::{from_raw_parts, from_raw_parts_mut};

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

#[no_mangle]
pub extern "C" fn init_X(layer_count: usize, input_len: usize, hidden_len : usize) -> Vec<Vec<f32>>{
    let mut x:Vec<Vec<f32>> = Vec::with_capacity(layer_count);
    let input_with_bias = input_len + 1;
    let hidden_with_bias = hidden_len + 1;

    x[0] = Vec::with_capacity(input_with_bias);
    x[0][0] = 1.0;

    for l in 1..layer_count{
        x[l] = Vec::with_capacity(hidden_with_bias);
        x[l][0] = 1.0;
    }

    x
}

#[no_mangle]
pub extern "C" fn init_W(layer_count: usize, input_len: usize, hidden_len: usize, output_len: usize) -> Vec<Vec<Vec<f32>>>{
    let mut w:Vec<Vec<Vec<f32>>> = Vec::with_capacity(layer_count);
    let input_with_bias = input_len + 1;
    let hidden_with_bias = hidden_len + 1;


    for l in 0..layer_count{
        // i prend la valeur de la longueur de la couche x suivante
        let size_i = if l == layer_count - 1 {output_len} else {hidden_with_bias};
        w[l] = Vec::with_capacity(size_i);

        for i in 0..size_i{
            // j prend la valeur de la longueur de la couche x précédente
            let size_j = if l == 0 {input_len} else {hidden_with_bias};
            w[l][i] = Vec::with_capacity(size_j);

            //init j
            randomize_weights(&mut w[l][i]);
        }
    }
    w
}

#[no_mangle]
pub extern "C" fn init_Deltas(layer_count: usize, hidden_len: usize, output_len: usize) -> Vec<Vec<f32>> {
    let mut deltas: Vec<Vec<f32>> = Vec::with_capacity(layer_count);
    let hidden_with_bias = hidden_len + 1;

     for l in 0..layer_count{
         let size_i = if l == layer_count - 1 {output_len} else {hidden_with_bias};
         deltas[l] = Vec::with_capacity(size_i);
     }

    deltas
}

#[no_mangle]
pub extern "C" fn randomize_weights(vec: &mut Vec<f32>){
    let mut rng = rand::thread_rng();
    for i in 0..vec.len(){
        vec[i] = rng.gen_range(-1.0..1.0);
    }
}

#[no_mangle]
pub extern "C" fn create_NeuralNet(layer_count: usize, input_len: usize, hidden_len : usize, output_len : usize) -> *mut NeuralNet{
    let mut nn = NeuralNet{
        layer_count,
        input_len,
        hidden_len,
        output_len,
        x: init_X(layer_count, input_len, hidden_len),
        w: init_W(layer_count, input_len, hidden_len, output_len),
        out: Vec::with_capacity(output_len),
        deltas: init_Deltas(layer_count, hidden_len, output_len)
    };

    let boxed_nn = Box::new(nn);
    let ref_to_nn = Box::leak(boxed_nn);

    ref_to_nn
}

#[no_mangle]
pub extern "C" fn train_NeuralNet(nn_raw: *mut NeuralNet, x: *mut f32, x_length: usize, y: *mut f32, y_length: usize, sample_count: usize, epoch: usize, alpha: f32, is_classif: bool){
    let nn = unsafe {
        nn_raw.as_mut().unwrap()
    };
    let inputs = unsafe {from_raw_parts_mut(x, x_length)};
    let expected = unsafe {from_raw_parts_mut(y, y_length)};
    let sample_len = y_length / sample_count;

    for e in 0..epoch{
        for i in 0..sample_count{
            //fit inputs in the first row of x
            for data in 0..nn.input_len{
                nn.x[0][data] = inputs[(e * sample_len) + data];
            }

            feed_Forward(nn);

            let mut sample: Vec<f32> = Vec::with_capacity(sample_len);
            fill_vec(&mut expected[i*sample_len], &mut sample, sample_len);

            if is_classif{
                backprop_classification(nn, &sample, alpha);
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn feed_Forward(nn: &mut NeuralNet){

    for l in 0..nn.layer_count-1{
        for i in 0..nn.w[l].len() {
            let mut w_sum: f32 = 0.0;

            for j in 0..nn.w[l][i].len() {
                w_sum += nn.w[l][i][j] * nn.x[l][j];
            }

            nn.x[l+1][i] = w_sum.tanh();
        }
    }

    for o in 0..nn.output_len{
        let mut output = 0.0;
        let cur_l_idx = nn.layer_count-1;
        let cur_l_len = nn.x[cur_l_idx].len();
        for j in 0..cur_l_len{
            output += nn.w[cur_l_idx][o][j] * nn.x[cur_l_idx][j];
        }

        nn.out[o] = output.tanh();
    }
}

#[no_mangle]
pub extern "C" fn backprop_classification(nn: &mut NeuralNet, y: &Vec<f32>, alpha: f32){
    let mut idx = nn.layer_count - 1;
    deltas_last_layer(&nn.out, &y, &mut nn.deltas[idx]);

    for l in 1..nn.layer_count{
        idx = nn.layer_count - l - 1;

        deltas_hidden_layer(&nn.x[idx], &nn.w[idx], &mut nn.deltas[idx+1]);
    }

    correct_w(nn, alpha);
}

#[no_mangle]
pub extern "C" fn correct_w(nn: &mut NeuralNet, alpha: f32){
    for l in 0..nn.layer_count{
        for i in 0..nn.w[l].len(){
            for j in 0..nn.w[l][i].len(){
                nn.w[l][i][j] = nn.w[l][i][j] - alpha * nn.x[l][j] * nn.deltas[l][i];
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn deltas_hidden_layer(x: &Vec<f32>, w: &Vec<Vec<f32>>, deltas: &mut Vec<f32>){
    for i in 0..deltas.len(){
        deltas[i] = (1.0 - x[i].powf(2.0)) * sum_deltas(&w[i], &deltas);
    }
}

#[no_mangle]
pub extern "C" fn deltas_last_layer(x: &Vec<f32>, y: &Vec<f32>, deltas: &mut Vec<f32>){
    for i in 0..deltas.len(){
        deltas[i] = (1.0 - x[i].powf(2.0)) * (x[i] - y[i]);
    }
}

#[no_mangle]
pub extern "C" fn sum_deltas(w: &Vec<f32>, deltas: &Vec<f32>) -> f32{
    deltas.iter().zip(w.iter()).map(|(x, y)| x * y).fold(0.0, |sum, i| sum + i)
}

#[no_mangle]
pub extern "C" fn fill_vec(origin_ptr: *mut f32, y: &mut Vec<f32>, len: usize) -> &Vec<f32>{
    let origin = unsafe {from_raw_parts_mut(origin_ptr, len)};
    for i in 0..len{
        y[i] = origin[i];
    }
    y
}

#[no_mangle]
pub extern "C" fn release_NeuralNet(nn: *mut NeuralNet){
    unsafe {
        Box::from_raw(nn);
    }
}