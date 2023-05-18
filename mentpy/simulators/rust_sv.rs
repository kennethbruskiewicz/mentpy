// Trying to write a rust SV simulator translating the code
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::PyString;
use pyo3::types::IntoPyDict;
use ndarray::Array2;
use num_complex::Complex;
use std::collections::HashMap;


pub struct MBQCircuit {
    py_mbqcircuit: PyObject,
}

#[pymethods]
pub struct NumpySimulatorSV {
    mbqcircuit: MBQCircuit, // this type will have to be defined
    input_state: Array2<Complex<f64>>,
    window_size: usize,
    schedule: Option<String>, // replace with the appropriate type
    force0: bool,
}

impl MBQCircuit {
    pub fn new(py: Python, py_mbqcircuit: PyObject) -> PyResult<Self> {
        // Here you could add any checks to make sure the PyObject is actually an instance of the MBQCircuit Python class.
        Ok(MBQCircuit {
            py_mbqcircuit,
        })
    }

    // Then, you can add methods to call Python methods on the MBQCircuit class.
    // For example, if MBQCircuit has a method 'get_graph':
    pub fn get_graph(&self, py: Python) -> PyResult<PyObject> {
        let get_graph: Py<PyAny> = self.py_mbqcircuit.call_method1(py, "get_graph", ())?;
        Ok(get_graph.into())
    }
}

impl NumpySimulatorSV {
    #[new]
    #[args(kwargs = "**")]
    pub fn new(mbqcircuit: MBQCircuit, input_state: Option<Array2<Complex<f64>>>, kwargs: Option<&PyDict>) -> PyResult<Self> {
        let mut window_size = 1;
        let mut schedule = None;
        let mut force0 = true;
        
        if let Some(kwargs) = kwargs {
            if let Some(value) = kwargs.get_item("window_size") {
                window_size = value.extract()?;
            }
            if let Some(value) = kwargs.get_item("schedule") {
                schedule = Some(value.extract()?);
            }
            if let Some(value) = kwargs.get_item("force0") {
                force0 = value.extract()?;
            }
        }

        if !force0 {
            return Err(PyErr::new::<NotImplementedError, _>("Numpy simulator does not support force0=False."));
        }

        Ok(NumpySimulatorSV {
            mbqcircuit,
            input_state: input_state.unwrap_or_else(|| Array2::from_elem((1, 1), Complex::new(1.0, 0.0))),
            window_size,
            schedule,
            force0,
        })
    }
}
