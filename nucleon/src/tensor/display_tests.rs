use crate::tensor::Tensor;
#[test]
fn display_prints_shape_header_and_matrix_rows() {
    // exact output, trailing newline after the last row included
    let t = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(format!("{t}"), "Tensor(shape=[2, 2])\n[1 2]\n[3 4]\n");
}

#[test]
fn display_header_disambiguates_same_looking_rows() {
    // scalar, [1], and [1,1] all render the row "[7]"; the header differs
    let scalar = Tensor::new(vec![], vec![7.0]);
    let vector = Tensor::new(vec![1], vec![7.0]);
    let matrix = Tensor::new(vec![1, 1], vec![7.0]);
    assert_eq!(format!("{scalar}"), "Tensor(shape=[])\n[7]\n");
    assert_eq!(format!("{vector}"), "Tensor(shape=[1])\n[7]\n");
    assert_eq!(format!("{matrix}"), "Tensor(shape=[1, 1])\n[7]\n");
}

#[test]
fn display_summarizes_large_tensors() {
    // exact match: a regression that also dumped rows would still contain
    // the header, so substring-matching would hide it
    let t = Tensor::zeros(vec![100, 100]);
    assert_eq!(format!("{t}"), "Tensor(shape=[100, 100])");
}

#[test]
fn display_summarizes_empty_tensors() {
    // a zero-sized axis is constructible; the printer must not emit nothing
    let t = Tensor::zeros(vec![2, 0]);
    assert_eq!(format!("{t}"), "Tensor(shape=[2, 0])");
}
