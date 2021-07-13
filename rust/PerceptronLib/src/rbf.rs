use std::slice::{from_raw_parts_mut, from_raw_parts};
use rand::Rng;

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
                let distance = euclidian_distance(slice.as_mut_ptr(), clst_slice.as_mut_ptr(), ndim);
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
    // println!("{}", counter);
}

fn euclidian_distance(p1_raw : *mut f32, p2_raw : *mut f32, ndim: usize) -> f32{
    let p1 = unsafe{from_raw_parts(p1_raw, ndim)};
    let p2 = unsafe{from_raw_parts(p2_raw, ndim)};

    let sum: f32 = p1.iter()
        .zip(p2.into_iter())
        .map(|(x, y)| (x - y).powf(2.0))
        .fold(0.0, |sum, i| sum + i);

    sum.sqrt()
}