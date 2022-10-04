use ndarray::Dim;
use numpy::PyArray;
use pyo3::{prelude::*, pymodule};

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyclass]
#[derive(Clone)]
pub struct DataParameters {
    parameters: audio_samples::DataParameters,
}

#[pymethods]
impl DataParameters {
    #[new]
    #[args(
        min_frequency = "20.",
        max_frequency = "20000.",
        sample_rate = "44100",
        num_samples = "256"
    )]
    fn new(sample_rate: u32, min_frequency: f32, max_frequency: f32, num_samples: u64) -> Self {
        Self {
            parameters: audio_samples::DataParameters::new(
                sample_rate,
                (min_frequency, max_frequency),
                num_samples,
            ),
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct Audio {
    samples: Vec<f32>,
    sample_rate: u32,
}

#[pymethods]
impl Audio {
    fn get_sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn get_samples<'py>(&self, py: Python<'py>) -> &'py PyArray<f32, Dim<[usize; 1]>> {
        PyArray::from_vec(py, self.samples.clone())
    }
}

impl From<audio_samples::Audio> for Audio {
    fn from(audio: audio_samples::Audio) -> Self {
        Self {
            samples: audio.samples,
            sample_rate: audio.sample_rate,
        }
    }
}
#[pyclass]
pub struct DataGenerator {
    generator: audio_samples::DataGenerator,
}

#[pymethods]
impl DataGenerator {
    #[new]
    fn new(data_parameters: DataParameters) -> Self {
        let data_parameters = data_parameters.parameters;
        Self {
            generator: audio_samples::DataGenerator::new(data_parameters, 0),
        }
    }

    fn next(&mut self) -> Audio {
        self.generator
            .next()
            .unwrap()
            .unwrap()
            .audio()
            .clone()
            .into()
    }

    fn next_n(&mut self, num_data_points: u32) -> Vec<Audio> {
        let mut result = Vec::with_capacity(num_data_points as usize);
        for _ in 0..num_data_points {
            result.push(self.next());
        }
        result
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn audio_samples_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Audio>()?;
    m.add_class::<DataParameters>()?;
    m.add_class::<DataGenerator>()?;
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}
