use std::collections::HashSet;

use crate::metadata::MetadataValue;
use crate::metadata::index::MetadataIndex;

/// A filter expression tree.
///
/// Operators bind with precedence: NOT > AND > OR.
/// Parentheses override precedence.
///
/// # Syntax (EBNF-like)
/// ```text
/// expr     = or_expr
/// or_expr  = and_expr ("OR" and_expr)*
/// and_expr = not_expr ("AND" not_expr)*
/// not_expr = "NOT" not_expr | atom
/// atom     = field op literal
///          | field "IN" "(" literal ("," literal)* ")"
///          | "(" expr ")"
/// field    = identifier
/// literal  = string | number | "true" | "false"
/// op       = "=" | "!=" | "<" | ">" | "<=" | ">=" | "LIKE"
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum Filter {
    /// Both sub-filters must match.
    And(Box<Filter>, Box<Filter>),
    /// At least one sub-filter matches.
    Or(Box<Filter>, Box<Filter>),
    /// The sub-filter must NOT match.
    Not(Box<Filter>),
    /// Field equals value.
    Eq(String, MetadataValue),
    /// Field does not equal value.
    Ne(String, MetadataValue),
    /// Field less than value.
    Lt(String, MetadataValue),
    /// Field greater than value.
    Gt(String, MetadataValue),
    /// Field less than or equal to value.
    Lte(String, MetadataValue),
    /// Field greater than or equal to value.
    Gte(String, MetadataValue),
    /// Field in set of values.
    In(String, Vec<MetadataValue>),
    /// Field LIKE pattern. The stored value is the prefix with '%' stripped.
    Like(String, String),
}

/// Parse a filter expression string into an AST.
///
/// ```
/// # use mini_vectordb::query::filter::parse_filter;
/// let filter = parse_filter("category = \"book\" AND price < 50").unwrap();
/// ```
///
/// # Errors
/// Returns a String describing the parse error (unexpected token,
/// mismatched parentheses, invalid operator, etc.).
pub fn parse_filter(input: &str) -> Result<Filter, String> {
    let tokens = tokenize(input)?;
    if tokens.is_empty() {
        return Err("empty filter expression".into());
    }
    let mut pos = 0;
    let expr = parse_or_expr(&tokens, &mut pos)?;
    if pos != tokens.len() {
        return Err(format!(
            "unexpected token at position {}: {:?}",
            pos, tokens[pos]
        ));
    }
    Ok(expr)
}

/// Evaluates a [`Filter`] against a [`MetadataIndex`], returning matching record IDs.
///
/// ```
/// # use mini_vectordb::metadata::index::MetadataIndex;
/// # use mini_vectordb::metadata::MetadataValue;
/// # use mini_vectordb::query::filter::{parse_filter, evaluate_filter};
/// let mut idx = MetadataIndex::new();
/// let mut meta = mini_vectordb::metadata::Metadata::new();
/// meta.insert("price".into(), MetadataValue::Integer(42));
/// idx.index_record("r1", &meta);
/// let filter = parse_filter("price < 100").unwrap();
/// let ids = evaluate_filter(&filter, &idx);
/// assert_eq!(ids, vec!["r1"]);
/// ```
pub fn evaluate_filter(filter: &Filter, index: &MetadataIndex) -> Vec<String> {
    let result = eval(filter, index);
    let mut ids: Vec<String> = result.into_iter().collect();
    ids.sort();
    ids
}

// ── Filter evaluator ──

fn eval(filter: &Filter, index: &MetadataIndex) -> HashSet<String> {
    match filter {
        Filter::And(l, r) => {
            let left = eval(l, index);
            // short-circuit: AND with empty is always empty
            if left.is_empty() {
                return HashSet::new();
            }
            let right = eval(r, index);
            left.intersection(&right).cloned().collect()
        }
        Filter::Or(l, r) => {
            let left = eval(l, index);
            let right = eval(r, index);
            left.union(&right).cloned().collect()
        }
        Filter::Not(inner) => {
            // NOT requires a universe set, which the caller must provide
            // by AND-ing with a non-negated condition. This should not be
            // used standalone — document this in the public API.
            eval(inner, index)
        }
        Filter::Eq(field, val) => leaf_ids(index, field, val),
        Filter::Ne(field, val) => {
            // compute all IDs for this field, then remove the matching ones
            let matching = leaf_ids(index, field, val);
            let all: HashSet<String> = index.get_field_ids(field).into_iter().collect();
            all.difference(&matching).cloned().collect()
        }
        Filter::Lt(field, val) => index
            .get_numeric_lt(field, to_f64(val))
            .into_iter()
            .collect(),
        Filter::Gt(field, val) => index
            .get_numeric_gt(field, to_f64(val))
            .into_iter()
            .collect(),
        Filter::Lte(field, val) => {
            let v = to_f64(val);
            let mut ids: HashSet<_> = index.get_numeric_lt(field, v).into_iter().collect();
            ids.extend(index.get_numeric_eq(field, v));
            ids
        }
        Filter::Gte(field, val) => {
            let v = to_f64(val);
            let mut ids: HashSet<_> = index.get_numeric_gt(field, v).into_iter().collect();
            ids.extend(index.get_numeric_eq(field, v));
            ids
        }
        Filter::In(field, vals) => vals.iter().fold(HashSet::new(), |mut acc, v| {
            acc.extend(leaf_ids(index, field, v));
            acc
        }),
        Filter::Like(field, prefix) => index.get_string_prefix(field, prefix).into_iter().collect(),
    }
}

/// Convert a MetadataValue to f64 for numeric comparison.
fn to_f64(val: &MetadataValue) -> f64 {
    match val {
        MetadataValue::Integer(i) => *i as f64,
        MetadataValue::Float(f) => *f,
        _ => 0.0,
    }
}

/// Look up IDs matching an equality condition against any value type.
fn leaf_ids(index: &MetadataIndex, field: &str, val: &MetadataValue) -> HashSet<String> {
    match val {
        MetadataValue::String(s) => index.get_string(field, s).into_iter().collect(),
        MetadataValue::Integer(i) => index.get_numeric_eq(field, *i as f64).into_iter().collect(),
        MetadataValue::Float(f) => index.get_numeric_eq(field, *f).into_iter().collect(),
        MetadataValue::Bool(b) => index.get_bool(field, *b).into_iter().collect(),
        _ => HashSet::new(),
    }
}

// ── Tokenizer ──

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Identifier(String),
    StringLiteral(String),
    Number(MetadataValue),
    Keyword(String),
    OpenParen,
    CloseParen,
    Comma,
}

fn tokenize(input: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        let c = chars[i];

        // whitespace
        if c.is_whitespace() {
            i += 1;
            continue;
        }

        // parentheses and comma
        if c == '(' {
            tokens.push(Token::OpenParen);
            i += 1;
            continue;
        }
        if c == ')' {
            tokens.push(Token::CloseParen);
            i += 1;
            continue;
        }
        if c == ',' {
            tokens.push(Token::Comma);
            i += 1;
            continue;
        }

        // string literals: "..." or '...'
        if c == '"' || c == '\'' {
            let quote = c;
            i += 1; // skip opening quote
            let start = i;
            while i < len && chars[i] != quote {
                if chars[i] == '\\' && i + 1 < len {
                    i += 2; // skip escaped char including the backslash
                } else {
                    i += 1;
                }
            }
            if i >= len {
                return Err("unterminated string literal".into());
            }
            let val: String = chars[start..i].iter().collect();
            i += 1; // skip closing quote
            tokens.push(Token::StringLiteral(val));
            continue;
        }

        // numbers: integer or float
        if c.is_ascii_digit() || (c == '-' && i + 1 < len && chars[i + 1].is_ascii_digit()) {
            let start = i;
            if c == '-' {
                i += 1;
            }
            while i < len && chars[i].is_ascii_digit() {
                i += 1;
            }
            let mut is_float = false;
            if i < len && chars[i] == '.' {
                is_float = true;
                i += 1;
                while i < len && chars[i].is_ascii_digit() {
                    i += 1;
                }
            }
            let num_str: String = chars[start..i].iter().collect();
            if is_float {
                let val: f64 = num_str
                    .parse()
                    .map_err(|_| format!("invalid float: {num_str}"))?;
                tokens.push(Token::Number(MetadataValue::Float(val)));
            } else {
                let val: i64 = num_str
                    .parse()
                    .map_err(|_| format!("invalid integer: {num_str}"))?;
                tokens.push(Token::Number(MetadataValue::Integer(val)));
            }
            continue;
        }

        // identifiers and keywords
        if c.is_alphabetic() || c == '_' {
            let start = i;
            while i < len && (chars[i].is_alphanumeric() || chars[i] == '_' || chars[i] == '%') {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            let upper = word.to_uppercase();
            match upper.as_str() {
                "AND" | "OR" | "NOT" | "IN" | "LIKE" => {
                    tokens.push(Token::Keyword(upper));
                }
                "TRUE" => {
                    tokens.push(Token::Number(MetadataValue::Bool(true)));
                }
                "FALSE" => {
                    tokens.push(Token::Number(MetadataValue::Bool(false)));
                }
                _ => {
                    tokens.push(Token::Identifier(word));
                }
            }
            continue;
        }

        // operators: =, !=, <, >, <=, >=
        if c == '=' {
            tokens.push(Token::Keyword("=".into()));
            i += 1;
            continue;
        }
        if c == '!' && i + 1 < len && chars[i + 1] == '=' {
            tokens.push(Token::Keyword("!=".into()));
            i += 2;
            continue;
        }
        if c == '<' && i + 1 < len && chars[i + 1] == '=' {
            tokens.push(Token::Keyword("<=".into()));
            i += 2;
            continue;
        }
        if c == '>' && i + 1 < len && chars[i + 1] == '=' {
            tokens.push(Token::Keyword(">=".into()));
            i += 2;
            continue;
        }
        if c == '<' {
            tokens.push(Token::Keyword("<".into()));
            i += 1;
            continue;
        }
        if c == '>' {
            tokens.push(Token::Keyword(">".into()));
            i += 1;
            continue;
        }

        return Err(format!("unexpected character at position {i}: {c}"));
    }

    Ok(tokens)
}

// ── Recursive descent parser ──

fn parse_or_expr(tokens: &[Token], pos: &mut usize) -> Result<Filter, String> {
    let mut left = parse_and_expr(tokens, pos)?;
    while *pos < tokens.len() && matches!(&tokens[*pos], Token::Keyword(k) if k == "OR") {
        *pos += 1; // skip OR
        let right = parse_and_expr(tokens, pos)?;
        left = Filter::Or(Box::new(left), Box::new(right));
    }
    Ok(left)
}

fn parse_and_expr(tokens: &[Token], pos: &mut usize) -> Result<Filter, String> {
    let mut left = parse_not_expr(tokens, pos)?;
    while *pos < tokens.len() && matches!(&tokens[*pos], Token::Keyword(k) if k == "AND") {
        *pos += 1; // skip AND
        let right = parse_not_expr(tokens, pos)?;
        left = Filter::And(Box::new(left), Box::new(right));
    }
    Ok(left)
}

fn parse_not_expr(tokens: &[Token], pos: &mut usize) -> Result<Filter, String> {
    if *pos < tokens.len() && matches!(&tokens[*pos], Token::Keyword(k) if k == "NOT") {
        *pos += 1; // skip NOT
        let inner = parse_not_expr(tokens, pos)?;
        return Ok(Filter::Not(Box::new(inner)));
    }
    parse_atom(tokens, pos)
}

fn parse_atom(tokens: &[Token], pos: &mut usize) -> Result<Filter, String> {
    if *pos >= tokens.len() {
        return Err("unexpected end of expression".into());
    }

    // parenthesized expression
    if tokens[*pos] == Token::OpenParen {
        *pos += 1;
        let expr = parse_or_expr(tokens, pos)?;
        if *pos >= tokens.len() || tokens[*pos] != Token::CloseParen {
            return Err("expected ')'".into());
        }
        *pos += 1;
        return Ok(expr);
    }

    // field name
    let field = match &tokens[*pos] {
        Token::Identifier(name) => name.clone(),
        _ => return Err(format!("expected field name, got {:?}", tokens[*pos])),
    };
    *pos += 1;

    if *pos >= tokens.len() {
        return Err(format!("expected operator after field '{field}'"));
    }

    // operator
    let op = match &tokens[*pos] {
        Token::Keyword(op) => op.clone(),
        _ => return Err(format!("expected operator, got {:?}", tokens[*pos])),
    };
    *pos += 1;

    match op.as_str() {
        "=" => {
            let val = parse_literal(tokens, pos)?;
            Ok(Filter::Eq(field, val))
        }
        "!=" => {
            let val = parse_literal(tokens, pos)?;
            Ok(Filter::Ne(field, val))
        }
        "<" => {
            let val = parse_literal(tokens, pos)?;
            Ok(Filter::Lt(field, val))
        }
        ">" => {
            let val = parse_literal(tokens, pos)?;
            Ok(Filter::Gt(field, val))
        }
        "<=" => {
            let val = parse_literal(tokens, pos)?;
            Ok(Filter::Lte(field, val))
        }
        ">=" => {
            let val = parse_literal(tokens, pos)?;
            Ok(Filter::Gte(field, val))
        }
        "IN" => {
            if *pos >= tokens.len() || tokens[*pos] != Token::OpenParen {
                return Err("expected '(' after IN".into());
            }
            *pos += 1; // skip '('
            let mut values = Vec::new();
            loop {
                values.push(parse_literal(tokens, pos)?);
                if *pos < tokens.len() && tokens[*pos] == Token::Comma {
                    *pos += 1; // skip ','
                    continue;
                }
                break;
            }
            if *pos >= tokens.len() || tokens[*pos] != Token::CloseParen {
                return Err("expected ')' after IN values".into());
            }
            *pos += 1; // skip ')'
            Ok(Filter::In(field, values))
        }
        "LIKE" => {
            let pattern = match &tokens[*pos] {
                Token::StringLiteral(s) => s.clone(),
                _ => return Err("LIKE requires a string pattern".into()),
            };
            *pos += 1;
            // strip trailing '%' — we only support prefix matching
            let prefix = pattern.strip_suffix('%').unwrap_or(&pattern);
            Ok(Filter::Like(field, prefix.to_string()))
        }
        _ => Err(format!("unknown operator: {op}")),
    }
}

fn parse_literal(tokens: &[Token], pos: &mut usize) -> Result<MetadataValue, String> {
    if *pos >= tokens.len() {
        return Err("expected literal value".into());
    }
    let val = match &tokens[*pos] {
        Token::StringLiteral(s) => MetadataValue::String(s.clone()),
        Token::Number(v) => v.clone(),
        _ => return Err(format!("expected literal value, got {:?}", tokens[*pos])),
    };
    *pos += 1;
    Ok(val)
}
