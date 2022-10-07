use audio_samples::{DataPointParameters, OscillatorTypeDistribution};
use ndarray::Dim;
use numpy::PyArray;
use pyo3::{prelude::*, pymodule};
use rand::distributions::Uniform;

#[pyfunction]
pub fn debug_txt() -> String {
    "0.1.6".to_string()
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
        num_samples = "256",
        seed_offset = "0"
    )]
    fn new(
        sample_rate: u32,
        min_frequency: f32,
        max_frequency: f32,
        num_samples: u64,
        seed_offset: u64,
    ) -> Self {
        Self {
            parameters: audio_samples::DataParameters::new(
                sample_rate,
                (min_frequency, max_frequency),
                num_samples,
                seed_offset,
            ),
        }
    }

    pub fn with_seed_offset(&self, seed_offset: u64) -> Self {
        Self {
            parameters: self.parameters.clone().with_seed_offset(seed_offset),
        }
    }

    pub fn add_sine(&self, amplitude_range: (f32, f32)) -> Self {
        DataParameters {
            parameters: self
                .parameters
                .clone()
                .with_oscillator(OscillatorTypeDistribution::Sine, amplitude_range),
        }
    }

    pub fn add_saw(&self, amplitude_range: (f32, f32)) -> Self {
        DataParameters {
            parameters: self
                .parameters
                .clone()
                .with_oscillator(OscillatorTypeDistribution::Saw, amplitude_range),
        }
    }

    pub fn add_pulse(&self, amplitude_range: (f32, f32), duty_cycle_range: (f32, f32)) -> Self {
        DataParameters {
            parameters: self.parameters.clone().with_oscillator(
                OscillatorTypeDistribution::Pulse(Uniform::new(
                    duty_cycle_range.0,
                    duty_cycle_range.1,
                )),
                amplitude_range,
            ),
        }
    }

    pub fn add_triangle(&self, amplitude_range: (f32, f32)) -> Self {
        DataParameters {
            parameters: self
                .parameters
                .clone()
                .with_oscillator(OscillatorTypeDistribution::Triangle, amplitude_range),
        }
    }

    pub fn add_noise(&self, amplitude_range: (f32, f32)) -> Self {
        DataParameters {
            parameters: self
                .parameters
                .clone()
                .with_oscillator(OscillatorTypeDistribution::Noise, amplitude_range),
        }
    }

    pub fn frequency_to_map(&self, frequency: f32) -> f32 {
        self.parameters.frequency_to_map(frequency)
    }

    pub fn map_to_frequency(&self, map: f32) -> f32 {
        self.parameters.map_to_frequency(map)
    }

    pub fn generate_at_index(&self, index: u64) -> DataPoint {
        self.parameters.generate(index).generate().unwrap().into()
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
#[derive(Clone)]
pub struct DataPoint {
    data: Audio,
    label: DataPointParameters,
}

#[pymethods]
impl DataPoint {
    fn get_audio(&self) -> Audio {
        self.data.clone()
    }

    fn get_samples<'py>(&self, py: Python<'py>) -> &'py PyArray<f32, Dim<[usize; 1]>> {
        self.data.get_samples(py)
    }

    fn get_frequency(&self) -> f32 {
        self.label.frequency
    }

    fn get_frequency_map(&self) -> f32 {
        self.label.frequency_map
    }
}

impl From<audio_samples::DataPoint> for DataPoint {
    fn from(data_point: audio_samples::DataPoint) -> Self {
        Self {
            data: data_point.audio.into(),
            label: data_point.label,
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
            generator: audio_samples::DataGenerator::new(data_parameters),
        }
    }

    fn next(&mut self) -> DataPoint {
        self.generator.next().unwrap().unwrap().clone().into()
    }

    fn next_n(&mut self, num_data_points: u32) -> Vec<DataPoint> {
        let mut result = Vec::with_capacity(num_data_points as usize);
        for _ in 0..num_data_points {
            result.push(self.next());
        }
        result
    }
}

impl Iterator for DataGenerator {
    type Item = DataPoint;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.next())
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn audio_samples_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Audio>()?;
    m.add_class::<DataPoint>()?;
    m.add_class::<DataParameters>()?;
    m.add_class::<DataGenerator>()?;
    m.add_function(wrap_pyfunction!(debug_txt, m)?)?;
    Ok(())
}
