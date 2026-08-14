use super::*;

// Hand-computed cases pin each op's definition; property tests catch what
// hand cases miss. CpuBackend is the reference every other backend is
// judged against, so these tests are the root of the whole trust chain.

fn t1(data: &[f32]) -> Tensor {
    Tensor::new(vec![data.len()], data.to_vec())
}

#[test]
fn add_and_mul_elementwise() {
    let cpu = CpuBackend;
    let a = t1(&[1.0, 2.0, 3.0]);
    let b = t1(&[10.0, 20.0, 30.0]);
    assert_eq!(cpu.add(&a, &b).data(), &[11.0, 22.0, 33.0]);
    assert_eq!(cpu.mul(&a, &b).data(), &[10.0, 40.0, 90.0]);
}

#[test]
fn silu_known_values() {
    let cpu = CpuBackend;
    // silu(0) = 0; silu(large) ~= large; silu(-large) ~= 0
    let out = cpu.silu(&t1(&[0.0, 20.0, -20.0]));
    assert_eq!(out.data()[0], 0.0);
    assert!((out.data()[1] - 20.0).abs() < 1e-3);
    assert!(out.data()[2].abs() < 1e-3);
}

#[test]
fn embed_returns_the_id_row() {
    let cpu = CpuBackend;
    let table = Tensor::from_fn(vec![5, 4], |i| (10 * i[0] + i[1]) as f32);
    assert_eq!(cpu.embed(&table, 3).data(), &[30.0, 31.0, 32.0, 33.0]);
}

#[test]
fn matvec_hand_case() {
    // [1 2; 3 4] @ [10, 100] = [210, 430], worked on paper
    let cpu = CpuBackend;
    let w = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let x = t1(&[10.0, 100.0]);
    assert_eq!(cpu.matvec(&w, &x).data(), &[210.0, 430.0]);
}

#[test]
fn matmul_hand_case() {
    // a = [1 2; 3 4] (2x2), b = [5 6; 7 8] (2x2, rows are dot-ready)
    // c[i][j] = a.row(i) . b.row(j) -> [[17, 23], [39, 53]]
    let cpu = CpuBackend;
    let a = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = Tensor::new(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    assert_eq!(cpu.matmul(&a, &b).data(), &[17.0, 23.0, 39.0, 53.0]);
}

#[test]
fn matmul_identity_returns_input() {
    let cpu = CpuBackend;
    let a = Tensor::from_fn(vec![3, 3], |i| (i[0] * 3 + i[1]) as f32);
    let id = Tensor::from_fn(vec![3, 3], |i| if i[0] == i[1] { 1.0 } else { 0.0 });
    assert_eq!(cpu.matmul(&a, &id).data(), a.data());
}

#[test]
fn rmsnorm_unit_rms_property() {
    // before the weight multiply, output must have RMS 1; use weight = 1
    let cpu = CpuBackend;
    let x = Tensor::from_fn(vec![64], |i| (i[0] as f32 - 32.0) / 7.0);
    let ones = Tensor::from_fn(vec![64], |_| 1.0);
    let out = cpu.rmsnorm(&x, &ones, 0.0);
    let rms = (out.data().iter().map(|v| v * v).sum::<f32>() / 64.0).sqrt();
    assert!((rms - 1.0).abs() < 1e-5, "rms was {rms}");
}

#[test]
fn softmax_sums_to_one_and_orders() {
    let cpu = CpuBackend;
    let out = cpu.softmax(&t1(&[1.0, 3.0, 2.0]));
    let sum: f32 = out.data().iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
    assert!(out.data()[1] > out.data()[2] && out.data()[2] > out.data()[0]);
}

#[test]
fn softmax_survives_huge_logits() {
    // without max subtraction exp(90) is inf and the output is NaN
    let cpu = CpuBackend;
    let out = cpu.softmax(&t1(&[90.0, 89.0]));
    assert!(out.data().iter().all(|v| v.is_finite()));
    assert!((out.data()[0] - 0.731).abs() < 1e-3);
}

#[test]
fn softmax_treats_neg_inf_as_masked() {
    let cpu = CpuBackend;
    let out = cpu.softmax(&t1(&[0.0, f32::NEG_INFINITY, 0.0]));
    assert_eq!(out.data()[1], 0.0);
    assert!((out.data()[0] - 0.5).abs() < 1e-6);
}

#[test]
fn rope_position_zero_is_identity() {
    let cpu = CpuBackend;
    let x = Tensor::from_fn(vec![2, 8], |i| (i[0] * 8 + i[1]) as f32);
    assert_eq!(cpu.rope(&x, 0, 1e6).data(), x.data());
}

#[test]
fn rope_hand_computed_pair() {
    // head_dim 2: one pair, freq = 1, angle = pos. [1, 0] at pos 1
    // rotates to [cos 1, sin 1]. Worked from the NeoX definition.
    let cpu = CpuBackend;
    let out = cpu.rope(&Tensor::new(vec![1, 2], vec![1.0, 0.0]), 1, 1e6);
    assert!((out.data()[0] - 1.0_f32.cos()).abs() < 1e-6);
    assert!((out.data()[1] - 1.0_f32.sin()).abs() < 1e-6);
}

#[test]
fn rope_preserves_pair_norms() {
    let cpu = CpuBackend;
    let x = Tensor::from_fn(vec![2, 8], |i| ((i[0] * 8 + i[1]) as f32).sin());
    let rotated = cpu.rope(&x, 12345, 1e6);
    for head in 0..2 {
        for i in 0..4 {
            let b = head * 8;
            let before = (x.at(&[head, i]).powi(2) + x.at(&[head, i + 4]).powi(2)).sqrt();
            let after = (rotated.data()[b + i].powi(2) + rotated.data()[b + i + 4].powi(2)).sqrt();
            assert!((before - after).abs() < 1e-5);
        }
    }
}

#[test]
fn attention_single_position_returns_that_value() {
    // seq of 1: softmax over one score is 1.0, output IS the value row
    let cpu = CpuBackend;
    let q = Tensor::new(vec![1, 4], vec![1.0, 0.0, 0.0, 0.0]);
    let k = Tensor::new(vec![1, 1, 4], vec![0.5, 0.5, 0.5, 0.5]);
    let v = Tensor::new(vec![1, 1, 4], vec![7.0, 8.0, 9.0, 10.0]);
    assert_eq!(cpu.attention(&q, &k, &v, 0.5).data(), v.data());
}

#[test]
fn attention_prefers_the_matching_key() {
    // q aligned with key 1 and orthogonal to key 0: output ~= value row 1
    let cpu = CpuBackend;
    let q = Tensor::new(vec![1, 2], vec![10.0, 0.0]);
    let k = Tensor::new(vec![1, 2, 2], vec![0.0, 10.0, 10.0, 0.0]);
    let v = Tensor::new(vec![1, 2, 2], vec![1.0, 1.0, 5.0, 5.0]);
    let out = cpu.attention(&q, &k, &v, 1.0);
    assert!((out.data()[0] - 5.0).abs() < 1e-3, "got {:?}", out.data());
}

#[test]
fn attention_gqa_heads_share_kv() {
    // 2 query heads, 1 kv head: both heads read the same k/v, and with
    // identical queries they must produce identical outputs
    let cpu = CpuBackend;
    let q = Tensor::new(vec![2, 2], vec![1.0, 2.0, 1.0, 2.0]);
    let k = Tensor::new(vec![1, 2, 2], vec![0.1, 0.2, 0.3, 0.4]);
    let v = Tensor::new(vec![1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let out = cpu.attention(&q, &k, &v, 1.0);
    assert_eq!(out.data()[..2], out.data()[2..]);
}

#[test]
fn argmax_ties_and_nan() {
    let cpu = CpuBackend;
    assert_eq!(cpu.argmax(&t1(&[1.0, 5.0, 5.0])), 1); // tie -> lowest
    assert_eq!(cpu.argmax(&t1(&[f32::NAN, -1.0])), 1); // NaN never wins
}

#[test]
fn attention_gqa_four_heads_two_kv_hand_case() {
    // heads 0,1 share kv 0; heads 2,3 share kv 1. seq=1 makes softmax a
    // no-op, so each head's output IS its kv head's value row — checked
    // by hand, with distinct values per kv head.
    let cpu = CpuBackend;
    let q = Tensor::from_fn(vec![4, 2], |i| (i[0] + 1) as f32);
    let k = Tensor::new(vec![2, 1, 2], vec![0.3, 0.4, 0.5, 0.6]);
    let v = Tensor::new(vec![2, 1, 2], vec![10.0, 11.0, 20.0, 21.0]);
    let out = cpu.attention(&q, &k, &v, 1.0);
    assert_eq!(
        out.data(),
        &[10.0, 11.0, 10.0, 11.0, 20.0, 21.0, 20.0, 21.0]
    );
}

#[test]
#[should_panic(expected = "caps seq at 4096")]
fn attention_rejects_seq_beyond_contract() {
    let cpu = CpuBackend;
    let q = Tensor::zeros(vec![1, 2]);
    let kv = Tensor::zeros(vec![1, 4097, 2]);
    cpu.attention(&q, &kv, &kv, 1.0);
}

#[test]
#[should_panic(expected = "nonzero")]
fn attention_rejects_empty_sequence() {
    let cpu = CpuBackend;
    let q = Tensor::zeros(vec![1, 2]);
    let kv = Tensor::zeros(vec![1, 0, 2]);
    cpu.attention(&q, &kv, &kv, 1.0);
}

#[test]
#[should_panic(expected = "nonzero")]
fn matvec_rejects_empty_input() {
    CpuBackend.matvec(&Tensor::zeros(vec![4, 0]), &Tensor::zeros(vec![0]));
}

#[test]
#[should_panic(expected = "nonzero")]
fn matmul_rejects_zero_inner_dim() {
    CpuBackend.matmul(&Tensor::zeros(vec![2, 0]), &Tensor::zeros(vec![3, 0]));
}

#[test]
#[should_panic(expected = "nonzero")]
fn rope_rejects_zero_head_dim() {
    CpuBackend.rope(&Tensor::zeros(vec![2, 0]), 1, 1e6);
}

#[test]
#[should_panic(expected = "1-D vector")]
fn matvec_rejects_non_vector_input() {
    // a [2,2] tensor must not silently flatten into a 4-vector
    CpuBackend.matvec(&Tensor::zeros(vec![4, 4]), &Tensor::zeros(vec![2, 2]));
}
