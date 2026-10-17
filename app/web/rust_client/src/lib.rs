use pyo3::prelude::*;
use serde_json::Value;

/// Fast JSON parser (production-grade: uses serde_json, faster than stdlib for large payloads)
#[pyfunction]
fn parse_json(data: &str) -> PyResult<Value> {
    let v: Value = serde_json::from_str(data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("JSON parse error: {}", e)))?;
    Ok(Python::with_gil(|py| {
        // Convert serde_json::Value to Python object via PyDict
        serde_json_to_py(&v, py)
    }))
}

fn serde_json_to_py(v: &Value, py: Python) -> PyObject {
    match v {
        Value::String(s) => s.to_object(py),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.to_object(py)
            } else if let Some(f) = n.as_f64() {
                f.to_object(py)
            } else {
                s.to_object(py)
            }
        }
        Value::Bool(b) => b.to_object(py),
        Value::Null => py.None(),
        Value::Array(arr) => {
            let list = pyo3::types::PyList::empty(py);
            for item in arr {
                list.append(serde_json_to_py(item, py)).unwrap();
            }
            list.into()
        }
        Value::Object(obj) => {
            let dict = pyo3::types::PyDict::new(py);
            for (k, val) in obj {
                dict.set_item(k, serde_json_to_py(val, py)).unwrap();
            }
            dict.into()
        }
    }
}

/// String processing helpers (C-level speed via Rust)
#[pyfunction]
fn lower_fast(input: &str) -> String {
    input.to_lowercase()
}

#[pyfunction]
fn contains_token(text: &str, token: &str) -> bool {
    text.contains(token)
}

#[pyfunction]
fn split_json_lines(text: &str) -> Vec<String> {
    text.lines().map(|line| line.to_string()).filter(|line| !line.is_empty()).collect()
}

#[pymodule]
fn rust_client(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_json, m)?)?;
    m.add_function(wrap_pyfunction!(lower_fast, m)?)?;
    m.add_function(wrap_pyfunction!(contains_token, m)?)?;
    m.add_function(wrap_pyfunction!(split_json_lines, m)?)?;
    m.add_class::<RustClient>()?;
    Ok(())
}

#[pyclass]
struct RustClient {
    server_url: String,
}

#[pymethods]
impl RustClient {
    #[new]
    fn new(url: &str) -> Self {
        RustClient { server_url: url.to_string() }
    }

    fn get_token_speed(&self, response_json: &str) -> PyResult<f64> {
        let v: Value = serde_json::from_str(response_json)
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Invalid JSON"))?;
        if let Some(obj) = v.as_object() {
            if let Some(tps) = obj.get("tokens_per_second").and_then(|val| val.as_f64()) {
                Ok(tps)
            } else {
                Ok(0.0)
            }
        } else {
            Ok(0.0)
        }
    }
}
