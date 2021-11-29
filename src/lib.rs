use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::collections::{HashMap, HashSet};
use std::mem;

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
fn forward_push_ppr_vec(
    edge_dict: HashMap<u32, Vec<u32>>,
    source: u32,
    damping_factor: f64,
    r_max: f64,
) -> PyResult<HashMap<u32, f64>> {
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
    let mut grown: HashSet<usize> = HashSet::with_capacity(edge_dict_len);
    let mut avail_mass: f64 = 1.0;
    r[source_index] = 1.0;
    grown.insert(source_index);

    while avail_mass > r_max {
        let grown_copy: HashSet<usize> = mem::take(&mut grown);
        for k in grown_copy {
            /*if avail_mass < r_max {
                break;
            }*/
            let res = mem::take(&mut r[k]);
            //r[k] = 0.0;

            let add_p = conversion_coefficient * res;
            p[k] += add_p;
            avail_mass -= add_p;

            let neighbourhood = &edge_list[k];
            let w = (damping_factor * res) / (neighbourhood.len() as f64);
            for &u in neighbourhood {
                // println!("{}", u);
                r[u] += w;
                grown.insert(u);
            }
        }
    }
    //let converted_p: HashMap<u32, f64> = p.iter().enumerate().map(|(x, y)| (index_to_name[x], *y)).collect();
    Ok(p.iter()
        .enumerate()
        .map(|(x, y)| (index_to_name[x], *y))
        .collect())
}

#[pyfunction]
fn forward_push_ppr_vec_lazy(
    edge_dict: HashMap<u32, Vec<u32>>,
    source: u32,
    damping_factor: f64,
    r_max: f64,
) -> PyResult<HashMap<u32, f64>> {
    let r_max = r_max.max(f64::EPSILON); // cap the r_max to epsilon

    let edge_dict_len = edge_dict.len().max(512);
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

    let mut grown: HashSet<usize> = HashSet::with_capacity(edge_dict_len);
    grown.insert(0);
    let mut avail_mass: f64 = 1.0;

    while avail_mass > r_max {
        let grown_cap = grown.capacity();
        let grown_copy: HashSet<usize> =
            mem::replace(&mut grown, HashSet::with_capacity(grown_cap));
        for k in grown_copy {
            let neighbourhood = &edge_list[k];
            let neighbourhood = match neighbourhood {
                None => {
                    update_edge_list(
                        k,
                        &edge_dict,
                        &mut index_to_name,
                        &mut name_to_index,
                        &mut edge_list,
                        &mut p,
                        &mut r,
                    );
                    edge_list[k].as_ref().unwrap()
                }
                Some(n) => n,
            };
            let res = mem::take(&mut r[k]);

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
    //let converted_p: HashMap<u32, f64> = p.iter().enumerate().map(|(x, y)| (index_to_name[x], *y)).collect();
    Ok(p.iter()
        .enumerate()
        .map(|(x, y)| (index_to_name[x], *y))
        .collect())
}

// function to update the bookkeeping dictionary and the other things in the lazy way
#[inline]
fn update_edge_list<'a>(
    node: usize,
    edge_dict: &HashMap<u32, Vec<u32>>,
    index_to_name: &mut Vec<u32>,
    name_to_index: &mut HashMap<u32, usize>,
    edge_list: &'a mut Vec<Option<Vec<usize>>>,
    p: &mut Vec<f64>,
    r: &mut Vec<f64>,
) -> () {
    //get name
    let node_name = index_to_name[node];

    let additional = edge_dict[&node_name].len();
    let mut neighbours = Vec::with_capacity(additional);
    // increases the size to avoid reallocating during the stuff
    name_to_index.reserve(additional);
    index_to_name.reserve(additional);
    edge_list.reserve(additional);
    p.reserve(additional);
    r.reserve(additional);

    for v_name in &edge_dict[&node_name] {
        if name_to_index.contains_key(v_name) {
            //no update necessary
            let v = name_to_index[v_name];
            neighbours.push(v);
        } else {
            let v = index_to_name.len();
            index_to_name.push(*v_name);
            name_to_index.insert(*v_name, v);
            neighbours.push(v);
            edge_list.push(Option::None);
            p.push(0.0);
            r.push(0.0);
        }
    }
    edge_list[node] = Option::Some(neighbours);
    // return &edge_list[node].expect("missing but just inserted");
}

#[pyfunction]
fn forward_push_ppr(
    edge_dict: HashMap<u32, Vec<u32>>,
    source: u32,
    damping_factor: f64,
    r_max: f64,
) -> PyResult<HashMap<u32, f64>> {
    let r_max = r_max.max(f64::EPSILON); // cap the r_max to epsilon
                                         //println!("{} {} {}", source, damping_factor, r_max);
    let conversion_coefficient = 1.0 - damping_factor;
    let mut p: HashMap<u32, f64> = HashMap::with_capacity(edge_dict.len());
    let mut r: HashMap<u32, f64> = HashMap::with_capacity(edge_dict.len());
    let mut grown: HashSet<u32> = HashSet::with_capacity(edge_dict.len());
    let mut avail_mass: f64 = 1.0;
    r.insert(source, 1.0);
    grown.insert(source);

    while avail_mass > r_max {
        let grown_copy: HashSet<u32> = grown.drain().collect();
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

/// A Python module implemented in Rust.
#[pymodule]
fn rustpyppr(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(forward_push_ppr, m)?)?;
    m.add_function(wrap_pyfunction!(forward_push_ppr_vec, m)?)?;
    m.add_function(wrap_pyfunction!(forward_push_ppr_vec_lazy, m)?)?;

    Ok(())
}
