use pyo3::prelude::*;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyclass]
pub struct Audio {
    samples: Vec<f32>,
    sample_rate: u32,
}

#[pymethods]
impl Audio {
    fn get_sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn get_samples(&self) -> Vec<f32> {
        self.samples.clone()
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
    fn new() -> Self {
        let data_parameters = audio_samples::DataParameters {
            sample_rate: 44100,
            num_samples: 256,
            frequency_range: (20., 20000.),
        };
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
    m.add_class::<DataGenerator>()?;
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}
