use super::*;

#[test]
fn display_names_the_stage() {
    let b = FamilyError::Build {
        reason: "x".to_string(),
    };
    let f = FamilyError::Forward {
        reason: "y".to_string(),
    };
    assert!(b.to_string().starts_with("family build failed"));
    assert!(f.to_string().starts_with("forward pass failed"));
}

#[test]
fn forward_constructor_wraps_any_display() {
    assert!(matches!(
        FamilyError::forward("boom"),
        FamilyError::Forward { .. }
    ));
}
