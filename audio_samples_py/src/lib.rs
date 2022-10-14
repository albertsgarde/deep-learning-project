use anyhow::Result;
use audio_samples::{
    data,
    parameters::{
        effects::EffectDistribution, oscillators::OscillatorTypeDistribution, DataPointParameters,
    },
};
use ndarray::Dim;
use numpy::PyArray;
use pyo3::{prelude::*, pymodule};
use rand::distributions::Uniform;

#[pyfunction]
pub fn debug_txt() -> String {
    "0.1.6".to_string()
}

/// Calculates the difference in cents between two frequencies.
#[pyfunction]
#[pyo3(text_signature = "(freq1, freq2, /)")]
pub fn cent_diff(freq1: f32, freq2: f32) -> f32 {
    audio_samples::cent_diff(freq1, freq2)
}

#[pyclass]
#[pyo3(
    text_signature = "(num_samples, sample_rate = 44100, min_frequency = 20, max_frequency=20000, /)"
)]
#[derive(Clone)]
pub struct DataParameters {
    parameters: audio_samples::parameters::DataParameters,
}

#[pymethods]
impl DataParameters {
    /// Create a new DataParameters object with no oscillators or effects.
    #[new]
    #[args(min_frequency = "20.", max_frequency = "20000.", sample_rate = "44100")]
    fn new(num_samples: u64, sample_rate: u32, min_frequency: f32, max_frequency: f32) -> Self {
        Self {
            parameters: audio_samples::parameters::DataParameters::new(
                sample_rate,
                (min_frequency, max_frequency),
                num_samples,
            ),
        }
    }

    /// Create a new DataParameters object with the given seed_offset.
    /// This is useful if you want to create multiple independent streams of data.
    #[pyo3(text_signature = "(self, seed_offset, /)")]
    pub fn with_seed_offset(&self, seed_offset: u64) -> Self {
        Self {
            parameters: self.parameters.clone().with_seed_offset(seed_offset),
        }
    }

    /// Adds a sine oscillator with the given amplitude range.
    #[pyo3(text_signature = "(self, amplitude_range, /)")]
    pub fn add_sine(&self, amplitude_range: (f32, f32)) -> Self {
        DataParameters {
            parameters: self
                .parameters
                .clone()
                .with_oscillator(OscillatorTypeDistribution::Sine, amplitude_range),
        }
    }

    /// Adds a saw oscillator with the given amplitude range.
    #[pyo3(text_signature = "(self, amplitude_range, /)")]
    pub fn add_saw(&self, amplitude_range: (f32, f32)) -> Self {
        DataParameters {
            parameters: self
                .parameters
                .clone()
                .with_oscillator(OscillatorTypeDistribution::Saw, amplitude_range),
        }
    }

    /// Adds a pulse oscillator with the given amplitude range and duty cycle.
    /// The duty cycle is the ratio of the pulse width to the period.
    #[pyo3(text_signature = "(self, amplitude_range, duty_cycle_range, /)")]
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

    /// Adds a triangle oscillator with the given amplitude range.
    #[pyo3(text_signature = "(self, amplitude_range, /)")]
    pub fn add_triangle(&self, amplitude_range: (f32, f32)) -> Self {
        DataParameters {
            parameters: self
                .parameters
                .clone()
                .with_oscillator(OscillatorTypeDistribution::Triangle, amplitude_range),
        }
    }

    /// Adds a noise oscillator with the given amplitude range.
    /// This will create samples that are normally distributed around `0.0` with the amplitude as standard deviation and any samples greater than the amplitude being cut off.
    #[pyo3(text_signature = "(self, amplitude_range, /)")]
    pub fn add_noise(&self, amplitude_range: (f32, f32)) -> Self {
        DataParameters {
            parameters: self
                .parameters
                .clone()
                .with_oscillator(OscillatorTypeDistribution::Noise, amplitude_range),
        }
    }

    /// Adds a distortion effect to all samples.
    /// The `power_range` argument determines the strength of the distortion.
    #[args(power_range = "(1., 1.)")]
    #[pyo3(text_signature = "(self, power_range, /)")]
    pub fn apply_distortion(&self, power_range: (f32, f32)) -> Self {
        let effect = EffectDistribution::distortion(power_range);
        DataParameters {
            parameters: self.parameters.clone().with_effect(effect),
        }
    }

    /// Given a frequency, returns the corresponding frequency mapping.
    #[pyo3(text_signature = "(self, frequency, /)")]
    pub fn frequency_to_map(&self, frequency: f32) -> f32 {
        self.parameters.frequency_to_map(frequency)
    }

    /// Given a frequency map value, returns the corresponding frequency.
    #[pyo3(text_signature = "(self, map, /)")]
    pub fn map_to_frequency(&self, map: f32) -> f32 {
        self.parameters.map_to_frequency(map)
    }

    /// Generates a samples at the given index.
    /// Calling this function multiple times with the same index will return the same samples.
    /// Calling this function multiple times with different indices will return (pseudo-)independent samples.
    #[pyo3(text_signature = "(self, index, /)")]
    pub fn generate_at_index(&self, index: u64) -> DataPoint {
        self.parameters.generate(index).generate().unwrap().into()
    }
}

/// Represents an audio clip.
#[pyclass]
#[derive(Clone)]
pub struct Audio {
    audio: audio_samples::Audio,
}

#[pymethods]
impl Audio {
    /// Loads an audio clip from the given path.
    #[staticmethod]
    #[pyo3(text_signature = "(path, /)")]
    fn from_wav(path: &str) -> Result<Self> {
        audio_samples::Audio::from_wav(path).map(|audio| audio.into())
    }

    /// Returns the sample rate of the audio.
    #[pyo3(text_signature = "(self, /)")]
    fn sample_rate(&self) -> u32 {
        self.audio.sample_rate
    }

    /// A vector of the samples in the audio.
    #[pyo3(text_signature = "(self, /)")]
    fn samples<'py>(&self, py: Python<'py>) -> &'py PyArray<f32, Dim<[usize; 1]>> {
        PyArray::from_vec(py, self.audio.samples.clone())
    }

    /// Saves the audio to a wav file.
    #[pyo3(text_signature = "(self, path, /)")]
    fn to_wav(&self, path: &str) -> Result<()> {
        self.audio.to_wav(path)
    }
}

impl From<audio_samples::Audio> for Audio {
    fn from(audio: audio_samples::Audio) -> Self {
        Self { audio }
    }
}

/// Represents a data point with an audio clip and the parameters used to generate it.
#[pyclass]
#[derive(Clone)]
pub struct DataPoint {
    data: Audio,
    label: DataPointParameters,
}

#[pymethods]
impl DataPoint {
    /// The audio.
    #[pyo3(text_signature = "(self, /)")]
    fn audio(&self) -> Audio {
        self.data.clone()
    }

    /// A vector of the samples in the audio.
    #[pyo3(text_signature = "(self, /)")]
    fn samples<'py>(&self, py: Python<'py>) -> &'py PyArray<f32, Dim<[usize; 1]>> {
        self.data.samples(py)
    }

    /// The fundamental frequency of the audio.
    #[pyo3(text_signature = "(self, /)")]
    fn frequency(&self) -> f32 {
        self.label.frequency
    }

    /// The fundamental frequency of the audio mapped into the range `[-1;1]`.
    #[pyo3(text_signature = "(self, /)")]
    fn frequency_map(&self) -> f32 {
        self.label.frequency_map
    }

    /// Saves the audio to a wav file.
    #[pyo3(text_signature = "(self, path, /)")]
    fn audio_to_wav(&self, path: &str) -> Result<()> {
        self.data.to_wav(path)
    }
}

impl From<data::DataPoint> for DataPoint {
    fn from(data_point: data::DataPoint) -> Self {
        Self {
            data: data_point.audio.into(),
            label: data_point.label,
        }
    }
}

#[pyclass]
pub struct DataGenerator {
    generator: data::DataGenerator,
}

#[pymethods]
impl DataGenerator {
    #[new]
    fn new(data_parameters: DataParameters) -> Self {
        let data_parameters = data_parameters.parameters;
        Self {
            generator: data::DataGenerator::new(data_parameters),
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
    m.add_function(wrap_pyfunction!(cent_diff, m)?)?;
    Ok(())
}
