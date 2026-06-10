// Filter AST parsing tests.

use mini_vectordb::metadata::MetadataValue;
use mini_vectordb::query::filter::{Filter, parse_filter};

fn s(v: &str) -> MetadataValue {
    MetadataValue::String(v.into())
}

fn i(v: i64) -> MetadataValue {
    MetadataValue::Integer(v)
}

fn b(v: bool) -> MetadataValue {
    MetadataValue::Bool(v)
}

// ── basic operators ──

#[test]
fn parse_equality_string() {
    let f = parse_filter("category = \"book\"").unwrap();
    assert_eq!(f, Filter::Eq("category".into(), s("book")));
}

#[test]
fn parse_equality_integer() {
    let f = parse_filter("price = 42").unwrap();
    assert_eq!(f, Filter::Eq("price".into(), i(42)));
}

#[test]
fn parse_equality_bool() {
    let f = parse_filter("active = true").unwrap();
    assert_eq!(f, Filter::Eq("active".into(), b(true)));
}

#[test]
fn parse_not_equal() {
    let f = parse_filter("status != \"deleted\"").unwrap();
    assert_eq!(f, Filter::Ne("status".into(), s("deleted")));
}

#[test]
fn parse_less_than() {
    let f = parse_filter("price < 100").unwrap();
    assert_eq!(f, Filter::Lt("price".into(), i(100)));
}

#[test]
fn parse_greater_than() {
    let f = parse_filter("score > 0.5").unwrap();
    assert_eq!(f, Filter::Gt("score".into(), MetadataValue::Float(0.5)));
}

#[test]
fn parse_less_equal() {
    let f = parse_filter("age <= 30").unwrap();
    assert_eq!(f, Filter::Lte("age".into(), i(30)));
}

#[test]
fn parse_greater_equal() {
    let f = parse_filter("rank >= 5").unwrap();
    assert_eq!(f, Filter::Gte("rank".into(), i(5)));
}

// ── LIKE ──

#[test]
fn parse_like_prefix() {
    let f = parse_filter("name LIKE \"prefix%\"").unwrap();
    assert_eq!(f, Filter::Like("name".into(), "prefix".into()));
}

#[test]
fn parse_like_without_percent() {
    let f = parse_filter("name LIKE \"exact\"").unwrap();
    assert_eq!(f, Filter::Like("name".into(), "exact".into()));
}

// ── IN ──

#[test]
fn parse_in_values() {
    let f = parse_filter("tag IN (\"rust\", \"go\", \"zig\")").unwrap();
    assert_eq!(
        f,
        Filter::In("tag".into(), vec![s("rust"), s("go"), s("zig")])
    );
}

#[test]
fn parse_in_single_value() {
    let f = parse_filter("id IN (42)").unwrap();
    assert_eq!(f, Filter::In("id".into(), vec![i(42)]));
}

// ── boolean logic ──

#[test]
fn parse_and() {
    let f = parse_filter("a = 1 AND b = 2").unwrap();
    assert_eq!(
        f,
        Filter::And(
            Box::new(Filter::Eq("a".into(), i(1))),
            Box::new(Filter::Eq("b".into(), i(2))),
        )
    );
}

#[test]
fn parse_or() {
    let f = parse_filter("a = 1 OR b = 2").unwrap();
    assert_eq!(
        f,
        Filter::Or(
            Box::new(Filter::Eq("a".into(), i(1))),
            Box::new(Filter::Eq("b".into(), i(2))),
        )
    );
}

#[test]
fn parse_not() {
    let f = parse_filter("NOT active = true").unwrap();
    assert_eq!(
        f,
        Filter::Not(Box::new(Filter::Eq("active".into(), b(true))))
    );
}

#[test]
fn parse_not_not() {
    let f = parse_filter("NOT NOT flag = false").unwrap();
    assert_eq!(
        f,
        Filter::Not(Box::new(Filter::Not(Box::new(Filter::Eq(
            "flag".into(),
            b(false)
        )))))
    );
}

// ── precedence ──

#[test]
fn parse_and_binds_tighter_than_or() {
    let f = parse_filter("a = 1 OR b = 2 AND c = 3").unwrap();
    assert_eq!(
        f,
        Filter::Or(
            Box::new(Filter::Eq("a".into(), i(1))),
            Box::new(Filter::And(
                Box::new(Filter::Eq("b".into(), i(2))),
                Box::new(Filter::Eq("c".into(), i(3))),
            ))
        )
    );
}

#[test]
fn parse_not_binds_tighter_than_and() {
    let f = parse_filter("NOT a = 1 AND b = 2").unwrap();
    assert_eq!(
        f,
        Filter::And(
            Box::new(Filter::Not(Box::new(Filter::Eq("a".into(), i(1))))),
            Box::new(Filter::Eq("b".into(), i(2))),
        )
    );
}

// ── parentheses ──

#[test]
fn parse_parentheses_override_precedence() {
    let f = parse_filter("(a = 1 OR b = 2) AND c = 3").unwrap();
    assert_eq!(
        f,
        Filter::And(
            Box::new(Filter::Or(
                Box::new(Filter::Eq("a".into(), i(1))),
                Box::new(Filter::Eq("b".into(), i(2))),
            )),
            Box::new(Filter::Eq("c".into(), i(3))),
        )
    );
}

#[test]
fn parse_nested_parentheses() {
    let f = parse_filter("((a = 1))").unwrap();
    assert_eq!(f, Filter::Eq("a".into(), i(1)));
}

// ── error cases ──

#[test]
fn parse_empty_input_returns_error() {
    assert!(parse_filter("").is_err());
    assert!(parse_filter("   ").is_err());
}

#[test]
fn parse_missing_operator_returns_error() {
    assert!(parse_filter("field").is_err());
}

#[test]
fn parse_missing_value_returns_error() {
    assert!(parse_filter("field =").is_err());
}

#[test]
fn parse_unmatched_paren_returns_error() {
    assert!(parse_filter("(a = 1").is_err());
}

#[test]
fn parse_invalid_operator_returns_error() {
    assert!(parse_filter("a + b").is_err());
}

#[test]
fn parse_trailing_tokens_returns_error() {
    assert!(parse_filter("a = 1 b = 2").is_err());
}

// ── complex expressions ──

#[test]
fn parse_complex_filter() {
    let f =
        parse_filter("category = \"book\" AND price < 50 OR category = \"film\" AND rating >= 4")
            .unwrap();
    // AND binds tighter: (book AND price<50) OR (film AND rating>=4)
    match &f {
        Filter::Or(left, right) => {
            assert!(matches!(**left, Filter::And(..)));
            assert!(matches!(**right, Filter::And(..)));
        }
        _ => panic!("expected Or at root"),
    }
}

#[test]
fn parse_negative_integer() {
    let f = parse_filter("x = -5").unwrap();
    assert_eq!(f, Filter::Eq("x".into(), i(-5)));
}

#[test]
fn parse_case_insensitive_keywords() {
    let f1 = parse_filter("a = 1 and b = 2").unwrap();
    let f2 = parse_filter("a = 1 AND b = 2").unwrap();
    assert_eq!(f1, f2);
}
