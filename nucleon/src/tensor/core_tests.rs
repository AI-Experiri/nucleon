use crate::tensor::Tensor;
#[test]
fn new_accepts_matching_shape_and_data() {
    let t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(t.shape(), &[2, 3]);
    assert_eq!(t.len(), 6);
}

#[test]
#[should_panic(expected = "wants 6 elements")]
fn new_panics_on_shape_data_mismatch() {
    Tensor::new(vec![2, 3], vec![1.0]);
}

#[test]
fn zeros_builds_zeroed_tensor() {
    let t = Tensor::zeros(vec![4, 2]);
    assert_eq!(t.len(), 8);
    assert!(t.data().iter().all(|&x| x == 0.0));
}

#[test]
fn from_fn_builds_from_indices() {
    // value = 10*i + j makes the expected layout obvious
    let t = Tensor::from_fn(vec![2, 3], |idx| (10 * idx[0] + idx[1]) as f32);
    assert_eq!(t.data(), &[0.0, 1.0, 2.0, 10.0, 11.0, 12.0]);
}

#[test]
fn rank_zero_scalar_works() {
    // shape [] has product 1: a scalar. at(&[]) is its only element.
    let t = Tensor::new(vec![], vec![7.0]);
    assert_eq!(t.at(&[]), 7.0);
    // same convention through the other constructors and the helper
    let u = Tensor::from_fn(vec![], |_| 3.0);
    assert_eq!(u.at(&[]), 3.0);
    assert_eq!(crate::tensor::checked_element_count(&[]), Some(1));
}

#[test]
#[should_panic(expected = "overflows")]
fn new_panics_on_shape_product_overflow() {
    // usize::MAX * 2 wraps under plain product(); the checked helper must
    // panic instead of accepting a tensor whose shape lies about its size
    Tensor::new(vec![usize::MAX, 2], vec![]);
}

#[test]
#[should_panic(expected = "overflows")]
fn zeros_panics_on_shape_product_overflow() {
    Tensor::zeros(vec![usize::MAX, 2]);
}

#[test]
#[should_panic(expected = "overflows")]
fn from_fn_panics_on_shape_product_overflow() {
    Tensor::from_fn(vec![usize::MAX, 2], |_| 0.0);
}

#[test]
fn zero_sized_axis_conventions() {
    // locked-down conventions for zero-sized axes, which backend loops
    // will rely on: rows of a [2,0] are empty slices, from_fn never calls
    // its closure, and len is 0
    let t = Tensor::zeros(vec![2, 0]);
    assert_eq!(t.row(0), &[] as &[f32]);
    assert_eq!(t.len(), 0);
    let u = Tensor::from_fn(vec![2, 0], |_| panic!("must not be called"));
    assert_eq!(u.len(), 0);
}

#[test]
fn checked_element_count_is_reusable_by_the_crate() {
    // the loader validates untrusted shapes with this exact helper
    assert_eq!(crate::tensor::checked_element_count(&[2, 3, 4]), Some(24));
    assert_eq!(crate::tensor::checked_element_count(&[usize::MAX, 2]), None);
}
