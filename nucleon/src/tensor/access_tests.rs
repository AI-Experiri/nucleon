use crate::tensor::Tensor;
#[test]
fn at_reads_row_major_order() {
    // [2,3] laid out as row 0 = [1,2,3], row 1 = [4,5,6]
    let t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(t.at(&[0, 0]), 1.0);
    assert_eq!(t.at(&[0, 2]), 3.0);
    assert_eq!(t.at(&[1, 0]), 4.0);
    assert_eq!(t.at(&[1, 2]), 6.0);
}

#[test]
#[should_panic(expected = "rank")]
fn at_panics_on_wrong_rank() {
    let t = Tensor::zeros(vec![2, 3]);
    t.at(&[1]);
}

#[test]
#[should_panic(expected = "out of bounds")]
fn at_panics_out_of_bounds() {
    let t = Tensor::zeros(vec![2, 3]);
    t.at(&[0, 3]);
}

#[test]
fn row_borrows_one_row_of_a_matrix() {
    let t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(t.row(0), &[1.0, 2.0, 3.0]);
    assert_eq!(t.row(1), &[4.0, 5.0, 6.0]);
}

#[test]
#[should_panic(expected = "2-D")]
fn row_panics_on_non_matrix() {
    let t = Tensor::zeros(vec![6]);
    t.row(0);
}

#[test]
fn at_reads_three_dims() {
    // value = 100*i + 10*j + k encodes the index, so layout errors show up
    let t = Tensor::from_fn(vec![2, 3, 4], |idx| {
        (100 * idx[0] + 10 * idx[1] + idx[2]) as f32
    });
    assert_eq!(t.at(&[0, 0, 0]), 0.0);
    assert_eq!(t.at(&[1, 2, 3]), 123.0);
    assert_eq!(t.at(&[0, 2, 1]), 21.0);
}

#[test]
#[should_panic(expected = "out of bounds")]
fn row_panics_out_of_bounds() {
    let t = Tensor::zeros(vec![2, 3]);
    t.row(2);
}
