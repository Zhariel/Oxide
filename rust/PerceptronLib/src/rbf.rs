use std::slice::{from_raw_parts_mut, from_raw_parts};
use crate::linear::{create_linear_model, train_rosenblatt_linear_model};
use rand::Rng;
use rand::seq::index::sample;

#[no_mangle]
pub extern "C" fn predict_rbf_classification(w_raw: *mut f32, x_raw: *mut f32, sample_raw: *mut f32, x_len: usize, ndim: usize, gamma: f32) -> f32 {
    let pred = predict_rbf_regression(w_raw, x_raw, sample_raw, x_len, ndim, gamma);
    let sign = if pred >= 0.0 {1.0} else {-1.0};

    sign
}

#[no_mangle]
pub extern "C" fn predict_rbf_regression(w_raw: *mut f32, x_raw: *mut f32, sample_raw: *mut f32, x_len: usize, ndim: usize, gamma: f32) -> f32{
    let sample_count = x_len / ndim;
    // let sample = unsafe{from_raw_parts(sample_raw, ndim)};
    // let w: Vec<f32> = vec![1.0; sample_count];
    let mut w = unsafe{from_raw_parts(w_raw, sample_count)};
    let x = unsafe{from_raw_parts_mut(x_raw, x_len)};
    let mut gauss_outputs: Vec<f32> = vec![0.0; sample_count];

    for i in 0..gauss_outputs.len(){
        let x_slice = x[i*ndim..(i+1)*ndim].as_mut_ptr();
        gauss_outputs[i] = (-gamma * norm(sample_raw, x_slice, ndim).powf(2.0)).exp()
    }

    let mut w_sum: f32 = 0.0;
    for i in 0..gauss_outputs.len(){
        w_sum += gauss_outputs[i] * w[i];
    }
    w_sum
}

fn norm(p1_raw : *mut f32, p2_raw : *mut f32, ndim: usize) -> f32{
    let p1 = unsafe{from_raw_parts(p1_raw, ndim)};
    let p2 = unsafe{from_raw_parts(p2_raw, ndim)};

    let mut sum: f32 = 0.0;
    for i in 0..ndim{
        sum += (p1[i] - p2[i]).powf(2.0);
    }
    sum.sqrt()
}

#[no_mangle]
pub extern "C" fn train_rosenblatt_rbf(w_raw: *mut f32, x_raw: *mut f32,  y_raw: *mut f32, x_len: usize, ndim: usize, iterations: usize, alpha: f32, gamma: f32) {
    let sample_count = x_len / ndim;
    let w = unsafe{from_raw_parts_mut(w_raw, sample_count)};
    let y = unsafe{from_raw_parts_mut(y_raw, sample_count)};
    let x = unsafe{from_raw_parts_mut(x_raw, x_len)};
    let mut rng = rand::thread_rng();

    let null_vec: *mut f32 = vec![0.0; ndim].as_mut_ptr();

    for i in 0..iterations{
        // let bias: *mut f32 = vec![1.0; ndim].as_mut_ptr();
        // let mut gxk = predict_rbf_regression(w_raw, x_raw, bias, x_len, ndim, gamma);
        // w[0] += alpha * (y[0] - gxk) * 1.0;

        for j in 0..sample_count{
            let xk_slice = &mut x[(j*ndim)..((j+1)*ndim)];
            let xk = xk_slice.as_mut_ptr();

            let gxk = predict_rbf_regression(w_raw, x_raw, xk, x_len, ndim, gamma);

            w[j] += alpha * (y[j] - gxk) * norm(xk, null_vec, ndim);
        }
    }
}

#[no_mangle]
pub extern "C" fn init_clusters(x: *mut f32, x_len: usize, nb_clusters: usize, ndim: usize) -> *mut f32{
    let inputs = unsafe {from_raw_parts_mut(x, x_len)};
    let mut clusters = vec![0.0; nb_clusters*ndim];
    let sample_count = x_len / ndim;
    let mut rng = rand::thread_rng();

    let mut min_vals = vec![f32::MAX; ndim];
    let mut max_vals = vec![f32::MIN; ndim];

    //Finding boundaries
    for i in 0..ndim{
        for j in 0..sample_count{
            if inputs[j*ndim+i] > max_vals[i]{max_vals[i] = inputs[j*ndim+i]};
            if inputs[j*ndim+i] < min_vals[i]{min_vals[i] = inputs[j*ndim+i]};
        }
    }

    let mut prev_indexes: Vec<usize> = vec![];

    for i in 0..nb_clusters{
        let mut idx: usize = rng.gen_range(0..sample_count);
        while prev_indexes.iter().any(|&i| i == idx){
            idx = rng.gen_range(0..sample_count);
        }
        prev_indexes.push(idx);

        for j in 0..ndim{
            clusters[i*ndim+j] = inputs[idx*ndim+j];
        }
    }

    let clusters_box = clusters.into_boxed_slice();
    let clusters_ref = Box::leak(clusters_box).as_mut_ptr();
    clusters_ref
}

#[no_mangle]
pub extern "C" fn k_means(x: *mut f32, x_len: usize, clusters_raw: *mut f32, nb_clusters: usize, ndim: usize){
    let inputs = unsafe {from_raw_parts_mut(x, x_len)};
    let clusters = unsafe {from_raw_parts_mut(clusters_raw, ndim*nb_clusters)};
    let sample_count = x_len / ndim;

    let mut complete = false;

    while !complete{
        complete = true;
        let mut assigned_clusters = vec![0; sample_count];

        //Assigning clusters
        for i in 0..sample_count {
            let mut closest_cluster = 0;
            let mut closest_dist = f32::MAX;

            let slice = &mut inputs[i * ndim..(i + 1) * ndim];

            for c in 0..nb_clusters {
                let clst_slice = &mut clusters[c * ndim..(c + 1) * ndim];
                let distance = norm(slice.as_mut_ptr(), clst_slice.as_mut_ptr(), ndim);
                if distance < closest_dist {
                    closest_cluster = c;
                    closest_dist = distance;
                }
            }

            assigned_clusters[i] = closest_cluster;
        }

        //Averaging clusters
        let mut coord_sum: Vec<f32> = vec![0.0; ndim*nb_clusters];
        for i in 0..sample_count {
            for j in 0..ndim {
                let cur_idx = assigned_clusters[i];
                coord_sum[cur_idx*ndim+j] += inputs[i*ndim+j];
            }
        }

        for i in 0..nb_clusters{
            let mut count = assigned_clusters.iter().filter(|&n| *n == i).count();
            count = if count == 0 { 1 } else { count };
            for j in 0..ndim{
                coord_sum[i*ndim+j] /= count as f32;
            }
        }

        //Assigning new values
        for i in 0..nb_clusters{
            for j in 0..ndim{
                let newval = coord_sum[i*ndim+j];
                if clusters[i*ndim+j] != newval{complete = false};
                clusters[i*ndim+j] = newval;
            }
        }
    }
}