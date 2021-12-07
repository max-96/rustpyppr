/*! This library provides functions for Python to compute [Personalized PageRank](https://arxiv.org/abs/2006.11876)
scores in a graph. Personalized PageRank is similar to ordinary [PageRank](https://en.wikipedia.org/wiki/PageRank)
but the node where the random surfer will start from (after a teleportation or in the beginning)
is not sampled uniformly from all the nodes, but rather from a smaller subset of them. This subset,
which is called Personalized Set, must contain at least 1 node.

The implemented algorithms so far are various versions of Forward Push for single source.
This means that given a source node, it is possible to compute the PPR scores of other ones w.r.t. the source node.

Extension to multiple sources is likely to happen soon.

!*/
use num_cpus;
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::collections::{HashMap, HashSet};
use std::mem;
use std::slice;
use std::sync::Arc;
use std::thread;

// Rust Implementation of Personalized Page Rank algorithms for Python 3

/*
    P = defaultdict(float)
    R = OrderedDict()
    R[source] = 1.0
    available_mass = 1.0

    while available_mass > 1e-4:
        v, r = R.popitem(False)
        gained_probability = (1 - damping_factor) * r
        P[v] += gained_probability
        available_mass -= gained_probability
        neighbourhood = graph[v]

        # if r / len(neighbourhood) > r_max:
        w = (damping_factor * r) / (len(neighbourhood))  # weight of the connection
        for u in neighbourhood:
            R[u] = R.get(u, 0) + w
            # R[u] += w

    return P
*/

#[pyfunction]
#[text_signature = "(edge_dict, sources, damping_factor, r_max)"]
/** Performs multiple forward push ppr on the same graph with different sources.

   The result is equivalent to calling the regular forward_push function sequentially.
   However, the results are computed in parallel. Performs parallel calls to [forward_push_vec].
   # Arguments
   * `edge_dict` - Dictionary mapping each node (positive integer) to the list of its neighbouring nodes (also positive integers).
   * `sources` - The list of nodes from which the PPR computations start.
   * `damping_factor` - The parameter that controls the probability of the random surfer to continue surfing. Typically 0.85.
   * `r_max` - controls the precision of the calculation. The computation will stop when at most `r_max` residual probability is left in total in the nodes.
   # Examples

   ```Python
   d = {3:[5, 1], 1:[3], 5:[3]}
   sources = [3, 5]
   ppr = multiple_forward_push_vec(d, sources, 0.85, 1e-2)
   ```
**/
pub fn multiple_forward_push_vec(
    edge_dict: HashMap<u32, Vec<u32>>,
    sources: Vec<u32>,
    damping_factor: f64,
    r_max: f64,
) -> PyResult<HashMap<u32, HashMap<u32, f64>>> {
    // make the edge_dict shareable across threads
    let edge_dict = Arc::new(edge_dict);
    let num_sources = sources.len();
    let mut join_handles = Vec::with_capacity(num_sources);
    let num_threads = num_cpus::get(); // number of threads to spawn

    let chunk_size = (num_sources as f32 / num_threads as f32).floor().max(1.0) as usize;
    let chunks: Vec<&[u32]> = sources.chunks(chunk_size).collect();
    for chunk in chunks {
        let ref_edge_dict = Arc::clone(&edge_dict);
        let chunk = chunk.to_vec();
        let handle = thread::spawn(move || {
            let mut results = Vec::with_capacity(chunk_size);
            for source in chunk {
                let p = _forward_push_vec(&ref_edge_dict, source, damping_factor, r_max);
                results.push((source, p));
            }
            results
        });
        join_handles.push(handle);
    }
    let mut pprs = HashMap::new();
    for handle in join_handles {
        let result = handle.join().unwrap();
        pprs.extend(result.into_iter());
    }
    Ok(pprs)
}

#[pyfunction]
#[text_signature = "(edge_dict, sources, damping_factor, r_max)"]
/**Performs multiple forward push ppr on the same graph with different sources.

The result is equivalent to calling the regular forward_push function sequentially.
However, the results are computed in parallel. Performs parallel calls to [forward_push_vec_lazy].
# Arguments
* `edge_dict` - Dictionary mapping each node (positive integer) to the list of its neighbouring nodes (also positive integers).
* `sources` - The list of nodes from which the PPR computations start.
* `damping_factor` - The parameter that controls the probability of the random surfer to continue surfing. Typically 0.85.
* `r_max` - controls the precision of the calculation. The computation will stop when at most `r_max` residual probability is left in total in the nodes.
# Examples

```Python
d = {3:[5, 1], 1:[3], 5:[3]}
sources = [3, 5]
ppr = multiple_forward_push_vec_lazy(d, sources, 0.85, 1e-2)
``` **/
pub fn multiple_forward_push_vec_lazy(
    edge_dict: HashMap<u32, Vec<u32>>,
    sources: Vec<u32>,
    damping_factor: f64,
    r_max: f64,
) -> PyResult<HashMap<u32, HashMap<u32, f64>>> {
    let edge_dict = Arc::new(edge_dict);
    let num_sources = sources.len();
    let mut join_handles = Vec::with_capacity(num_sources);
    let num_threads = num_cpus::get();
    let chunk_size = (num_sources as f32 / num_threads as f32).floor().max(1.0) as usize;
    let chunks: Vec<&[u32]> = sources.chunks(chunk_size).collect();
    for chunk in chunks {
        let ref_edge_dict = Arc::clone(&edge_dict);
        let chunk = chunk.to_vec();
        let handle = thread::spawn(move || {
            let mut results = Vec::with_capacity(chunk_size);
            for source in chunk {
                let p = _forward_push_vec_lazy(&ref_edge_dict, source, damping_factor, r_max);
                results.push((source, p));
            }
            results
        });
        join_handles.push(handle);
    }
    let mut pprs = HashMap::new();
    for handle in join_handles {
        let result = handle.join().unwrap();
        pprs.extend(result.into_iter());
    }
    Ok(pprs)
}

#[pyfunction]
#[text_signature = "(edge_dict, source, damping_factor, r_max)"]
/**Computes the Personalized PageRank using Forward Push, using vectors.

 Nodes that are visited are converted in indices, allowing fast lookup using a vector instead of HashMap.
 This conversion is done for all nodes eagerly. See [forward_push_vec_lazy] for the lazy version.
 # Arguments
 * `edge_dict` - Dictionary mapping each node (positive integer) to the list of its neighbouring nodes (also positive integers).
 * `source` - The node from which the PPR starts.
 * `damping_factor` - The parameter that controls the probability of the random surfer to continue surfing. Typically 0.85.
 * `r_max` - controls the precision of the calculation. The computation will stop when at most `r_max` residual probability is left in total in the nodes.
 # Examples

 ```Python
 d = {3:[5, 1], 1:[3], 5:[3]}
 source = 3
 ppr = forward_push_vec(d, source, 0.85, 1e-2)
 ```
**/
pub fn forward_push_vec(
    edge_dict: HashMap<u32, Vec<u32>>,
    source: u32,
    damping_factor: f64,
    r_max: f64,
) -> PyResult<HashMap<u32, f64>> {
    let sanity = check_arguments(&edge_dict, source, damping_factor, r_max);
    if sanity.is_some() {
        return Err(sanity.unwrap());
    }
    Ok(_forward_push_vec(&edge_dict, source, damping_factor, r_max))
}
// struct Buffer {
//     index: usize,
//     max_size: usize,
//     data: [(usize, f64); 32],
// }
//
// impl Buffer {
//     fn new() -> Buffer {
//         Buffer {
//             index: 0,
//             max_size: 32,
//             data: [(0, 0.0); 32],
//         }
//     }
// }

fn _forward_push_vec(
    edge_dict: &HashMap<u32, Vec<u32>>,
    source: u32,
    damping_factor: f64,
    r_max: f64,
) -> HashMap<u32, f64> {
    let r_max = r_max.max(f64::EPSILON); // cap the r_max to epsilon

    let edge_dict_len = edge_dict.len();
    // we first need to translate into vectors

    // give each node an index so to shrink the size
    let index_to_name: Vec<u32> = edge_dict.keys().map(|x| *x).collect();

    // build the inverse, from name to index
    let name_to_index: HashMap<u32, usize> = index_to_name
        .iter()
        .enumerate()
        .map(|(x, y)| (*y, x))
        .collect();

    let mut edge_list: Vec<Vec<usize>> = Vec::with_capacity(edge_dict_len);
    for k in &index_to_name {
        let v: Vec<usize> = edge_dict[k]
            .iter()
            .map(|name| name_to_index[name])
            .collect();
        edge_list.push(v);
    }
    let edge_list = edge_list;

    // Now let's start with the real algorithm
    let source_index: usize = name_to_index[&source];
    let conversion_coefficient = 1.0 - damping_factor;
    let mut p: Vec<f64> = vec![0.0; edge_dict_len];
    let mut r: Vec<f64> = vec![0.0; edge_dict_len];
    r[source_index] = 1.0;
    let mut grown = HashSet::new();
    grown.insert(source_index);
    let mut avail_mass: f64 = 1.0;

    while avail_mass > r_max {
        let grown_capacity = grown.capacity();
        let grown_copy: HashSet<usize> =
            mem::replace(&mut grown, HashSet::with_capacity(grown_capacity));
        for k in grown_copy {
            let res = mem::take(&mut r[k]);

            let add_p = conversion_coefficient * res;
            p[k] += add_p;
            avail_mass -= add_p;

            let neighbourhood = &edge_list[k];
            let w = (damping_factor * res) / (neighbourhood.len() as f64);
            for &u in neighbourhood {
                r[u] += w;
                grown.insert(u);
            }
        }
    }
    p.iter()
        .enumerate()
        .map(|(x, y)| (index_to_name[x], *y))
        .collect()
}

#[pyfunction]
#[text_signature = "(edge_dict, source, damping_factor, r_max)"]
/** Computes the Personalized PageRank using Forward Push, using vectors lazily.

 Nodes that are visited are converted in indices, allowing fast lookup using a vector instead of HashMap.
 This conversion is done for only for the nodes the algorithm encounters, in a lazy way.
 See [forward_push_vec] for the eager version.

 # Arguments
 * `edge_dict` - Dictionary mapping each node (positive integer) to the list of its neighbouring nodes (also positive integers).
 * `source` - The node from which the PPR starts.
 * `damping_factor` - The parameter that controls the probability of the random surfer to continue surfing. Typically 0.85.
 * `r_max` - controls the precision of the calculation. The computation will stop when at most `r_max` residual probability is left in total in the nodes.
 # Examples

 ```Python3
 d = {3:[5, 1], 1:[3], 5:[3]}
 source = 3
 ppr = forward_push_vec_lazy(d, source, 0.85, 1e-2)
 ```
**/
pub fn forward_push_vec_lazy(
    edge_dict: HashMap<u32, Vec<u32>>,
    source: u32,
    damping_factor: f64,
    r_max: f64,
) -> PyResult<HashMap<u32, f64>> {
    let sanity = check_arguments(&edge_dict, source, damping_factor, r_max);
    if sanity.is_some() {
        return Err(sanity.unwrap());
    }
    Ok(_forward_push_vec_lazy(
        &edge_dict,
        source,
        damping_factor,
        r_max,
    ))
}

#[inline]
fn _forward_push_vec_lazy(
    edge_dict: &HashMap<u32, Vec<u32>>,
    source: u32,
    damping_factor: f64,
    r_max: f64,
) -> HashMap<u32, f64> {
    let r_max = r_max.max(f64::EPSILON); // cap the r_max to epsilon

    let edge_dict_len = edge_dict.len();
    // we first need to translate into vectors

    // give each node an index so to shrink the size
    let mut index_to_name: Vec<u32> = Vec::with_capacity(edge_dict_len);
    index_to_name.push(source);

    // build the inverse, from name to index
    let mut name_to_index: HashMap<u32, usize> = HashMap::with_capacity(edge_dict_len);
    name_to_index.insert(source, 0);

    let mut edge_list: Vec<Option<Vec<usize>>> = Vec::with_capacity(edge_dict_len);
    edge_list.push(Option::None);

    // Now let's start with the real algorithm
    let mut p: Vec<f64> = Vec::with_capacity(edge_dict_len);
    p.push(0.0);
    let mut r: Vec<f64> = Vec::with_capacity(edge_dict_len);
    r.push(1.0);
    update_edge_list(
        0,
        &edge_dict,
        &mut index_to_name,
        &mut name_to_index,
        &mut edge_list,
        &mut p,
        &mut r,
    );

    let mut grown: HashSet<usize> = HashSet::new();
    grown.insert(0);
    let mut avail_mass: f64 = 1.0;

    while avail_mass > r_max {
        let grown_capacity = grown.capacity();
        let grown_copy: HashSet<usize> =
            mem::replace(&mut grown, HashSet::with_capacity(grown_capacity));
        for k in grown_copy {
            let neighbourhood = &edge_list[k];
            let neighbourhood = match neighbourhood {
                None => update_edge_list(
                    k,
                    &edge_dict,
                    &mut index_to_name,
                    &mut name_to_index,
                    &mut edge_list,
                    &mut p,
                    &mut r,
                ),
                Some(n) => n.as_slice(),
            };
            /*removes it from memory and replaces it with 0.0. Equivalent to
             * `let res = r[k]; r[k] = 0.0;` but faster.
             */
            let res = mem::take(&mut r[k]); //

            let add_p = (1.0 - damping_factor) * res;
            p[k] += add_p;
            avail_mass -= add_p;

            //let neighbourhood = &edge_list[k];
            let w = (damping_factor * res) / (neighbourhood.len() as f64);
            for &u in neighbourhood {
                // println!("{}", u);
                r[u] += w;
                grown.insert(u);
            }
        }
    }
    p.iter()
        .enumerate()
        .map(|(x, y)| (index_to_name[x], *y))
        .collect()
}

// function to update the bookkeeping dictionary and the other things in the lazy way
fn update_edge_list<'a>(
    node: usize,
    edge_dict: &HashMap<u32, Vec<u32>>,
    index_to_name: &mut Vec<u32>,
    name_to_index: &mut HashMap<u32, usize>,
    edge_list: &'a mut Vec<Option<Vec<usize>>>,
    p: &mut Vec<f64>,
    r: &mut Vec<f64>,
) -> &'a [usize] {
    //get name
    let node_name = index_to_name[node];
    // how many neighbours does the node have
    let additional = edge_dict[&node_name].len();
    let mut neighbours = Vec::with_capacity(additional);
    for &v_name in &edge_dict[&node_name] {
        if name_to_index.contains_key(&v_name) {
            // the neighbour is known
            //no update necessary
            let v = name_to_index[&v_name];
            neighbours.push(v);
        } else {
            //new neighbour, need to create the entry
            let v = index_to_name.len(); // the index of the neighbour
                                         //mapping the neighbour to its name
            index_to_name.push(v_name);
            name_to_index.insert(v_name, v);
            // saving the neighbour in the neighbours vector
            neighbours.push(v);
            // adding empty entries for the neighbour v
            edge_list.push(Option::None);
            p.push(0.0);
            r.push(0.0);
        }
    }
    /* Ideally, we would like to return a reference to the vector `neighbours`.
     * However, this is hard to do with the rules of rust, as we lose ownership of `neighbours`
     * when we insert it in `edge_list`. However, we know that in the `mother` function, this
     * reference will live for a shorter time than the reference to `edge_list`, since it will
     * be consumed in the for loop while the reference to edge_list will live longer. Also, does not
     * depend on the reallocation of edge_list in case it has too many elements. Therefore we return
     * an immutable slice of the neighbours.
     */
    let vec_pointer = neighbours.as_ptr();
    let length = neighbours.len();
    edge_list[node] = Some(neighbours);
    unsafe { slice::from_raw_parts(vec_pointer, length) }
}

#[pyfunction]
#[text_signature = "(edge_dict, source, damping_factor, r_max)"]
/** Computes the Personalized PageRank using Forward Push.

 In this version, no conversion from node to index takes place.
 For the lookup of the neighbours of a node, it uses `edge_dict` (HashMap).
 Although both HashMap and Vector lookups are **O(1)** in theory, the HashMap lookup is slower.
 This version is almost twice slower than the vector versions ([forward_push_vec], [forward_push_vec_lazy]).
 However, the vector takes extra space in memory that this version does not require.
 Thus, this version is only suggested when the vector versions fail due to memory limitations.

 # Arguments
 * `edge_dict` - Dictionary mapping each node (positive integer) to the list of its neighbouring nodes (also positive integers).
 * `source` - The node from which the PPR starts.
 * `damping_factor` - The parameter that controls the probability of the random surfer to continue surfing. Typically 0.85.
 * `r_max` - controls the precision of the calculation. The computation will stop when at most `r_max` residual probability is left in total in the nodes.
 # Examples

 ```Python
 d = {3:[5, 1], 1:[3], 5:[3]}
 source = 3
 ppr = forward_push(d, source, 0.85, 1e-2)
 ```
**/
pub fn forward_push(
    edge_dict: HashMap<u32, Vec<u32>>,
    source: u32,
    damping_factor: f64,
    r_max: f64,
) -> PyResult<HashMap<u32, f64>> {
    // sanity checks
    let sanity = check_arguments(&edge_dict, source, damping_factor, r_max);
    if sanity.is_some() {
        return Err(sanity.unwrap());
    }

    let r_max = r_max.max(f64::EPSILON); // cap the r_max to epsilon
                                         //println!("{} {} {}", source, damping_factor, r_max);
    let conversion_coefficient = 1.0 - damping_factor;
    let mut p: HashMap<u32, f64> = HashMap::with_capacity(edge_dict.len());
    let mut r: HashMap<u32, f64> = HashMap::with_capacity(edge_dict.len());
    let mut grown: HashSet<u32> = HashSet::new();
    let mut avail_mass: f64 = 1.0;
    r.insert(source, 1.0);
    grown.insert(source);

    while avail_mass > r_max {
        let grown_copy: HashSet<u32> = mem::take(&mut grown);
        for k in grown_copy {
            /*if avail_mass < r_max {
                return Ok(p);
            }*/
            let res = r.insert(k, 0.0).unwrap();
            let add_p = conversion_coefficient * res;
            *p.entry(k).or_insert(0.0) += add_p;
            avail_mass -= add_p;

            let neighbourhood = edge_dict.get(&k).unwrap();
            let w = (damping_factor * res) / (neighbourhood.len() as f64);
            for &u in neighbourhood {
                // println!("{}", u);
                *r.entry(u).or_insert(0.0) += w;
                grown.insert(u);
            }
        }
    }
    Ok(p)
}

fn check_arguments(
    edge_dict: &HashMap<u32, Vec<u32>>,
    source: u32,
    damping_factor: f64,
    r_max: f64,
) -> Option<PyErr> {
    if !edge_dict.contains_key(&source) {
        return Some(PyKeyError::new_err(format!(
            "source {} not found in edge_dict",
            source
        )));
    }
    if damping_factor >= 1.0 || damping_factor <= 0.0 {
        return Some(PyValueError::new_err(format!(
            "damping_factor {} not in bounds 0 < x < 1",
            damping_factor
        )));
    }
    if r_max >= 1.0 || r_max <= 0.0 {
        return Some(PyValueError::new_err(format!(
            "r_max {} not in bounds 0 < x < 1",
            r_max
        )));
    }
    None
}

#[pymodule]
fn rustpyppr(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(forward_push, m)?)?;
    m.add_function(wrap_pyfunction!(forward_push_vec, m)?)?;
    m.add_function(wrap_pyfunction!(forward_push_vec_lazy, m)?)?;
    m.add_function(wrap_pyfunction!(multiple_forward_push_vec, m)?)?;
    m.add_function(wrap_pyfunction!(multiple_forward_push_vec_lazy, m)?)?;

    Ok(())
}
