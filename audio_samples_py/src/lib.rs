use anyhow::Result;
use audio_samples::{
    data,
    parameters::{effects::EffectTypeDistribution, oscillators::OscillatorTypeDistribution},
};
use ndarray::Dim;
use numpy::PyArray;
use pyo3::{prelude::*, pymodule, types::PyType};
use rand::{distributions::Uniform, seq::SliceRandom};

#[pyfunction]
pub fn debug_txt() -> String {
    "0.1.8".to_string()
}

/// Calculates the difference in cents between two frequencies.
#[pyfunction]
#[pyo3(text_signature = "(freq1, freq2, /)")]
pub fn cent_diff(freq1: f32, freq2: f32) -> f32 {
    audio_samples::cent_diff(freq1, freq2)
}

/// Convert from frequency map to note number.
#[pyfunction]
#[pyo3(text_signature = "(map, /)")]
pub fn map_to_note_number(map: f32) -> f32 {
    audio_samples::map_to_note_number(map)
}

/// Convert from note number to frequency map.
#[pyfunction]
#[pyo3(text_signature = "(note_number, /)")]
pub fn note_number_to_map(note_number: f32) -> f32 {
    audio_samples::note_number_to_map(note_number)
}

/// Given a frequency, returns the corresponding frequency mapping.
#[pyfunction]
#[pyo3(text_signature = "(frequency, /)")]
pub fn frequency_to_map(frequency: f32) -> f32 {
    audio_samples::frequency_to_map(frequency)
}

/// Given a frequency map value, returns the corresponding frequency.
#[pyfunction]
#[pyo3(text_signature = "(map, /)")]
pub fn map_to_frequency(map: f32) -> f32 {
    audio_samples::map_to_frequency(map)
}

#[pyfunction]
#[pyo3(text_signature = "(/)")]
pub fn num_chord_types() -> usize {
    audio_samples::CHORD_TYPES.len()
}

#[pyfunction]
#[pyo3(text_signature = "(chord_type, /)")]
pub fn chord_type_name(chord_type: u32) -> String {
    audio_samples::CHORD_TYPES[chord_type as usize]
        .0
        .to_string()
}

#[pyclass]
#[pyo3(
    text_signature = "(num_samples, sample_rate = 44100, min_frequency = 20, max_frequency=20000, possible_chord_types=[0], /)"
)]
#[derive(Clone)]
pub struct DataParameters {
    parameters: audio_samples::parameters::DataParameters,
}

#[pymethods]
impl DataParameters {
    /// Create a new DataParameters object with no oscillators or effects.
    #[new]
    #[args(
        sample_rate = "44100",
        min_frequency = "20.",
        max_frequency = "20000.",
        possible_chord_types = "vec![0]"
    )]
    fn new(
        num_samples: u64,
        sample_rate: u32,
        min_frequency: f32,
        max_frequency: f32,
        possible_chord_types: Vec<u32>,
    ) -> Self {
        Self {
            parameters: audio_samples::parameters::DataParameters::new(
                sample_rate,
                (min_frequency, max_frequency),
                possible_chord_types,
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
    #[args(probability = "1.")]
    #[pyo3(text_signature = "(self, probability, amplitude_range, /)")]
    pub fn add_sine(&self, probability: f64, amplitude_range: (f32, f32)) -> Self {
        DataParameters {
            parameters: self.parameters.clone().with_oscillator(
                OscillatorTypeDistribution::Sine,
                probability,
                amplitude_range,
            ),
        }
    }

    /// Adds a saw oscillator with the given amplitude range.
    #[args(probability = "1.")]
    #[pyo3(text_signature = "(self, probability, amplitude_range, /)")]
    pub fn add_saw(&self, probability: f64, amplitude_range: (f32, f32)) -> Self {
        DataParameters {
            parameters: self.parameters.clone().with_oscillator(
                OscillatorTypeDistribution::Saw,
                probability,
                amplitude_range,
            ),
        }
    }

    /// Adds a pulse oscillator with the given amplitude range and duty cycle.
    /// The duty cycle is the ratio of the pulse width to the period.
    #[args(probability = "1.")]
    #[pyo3(text_signature = "(self, probability, amplitude_range, duty_cycle_range, /)")]
    pub fn add_pulse(
        &self,
        probability: f64,
        amplitude_range: (f32, f32),
        duty_cycle_range: (f32, f32),
    ) -> Self {
        DataParameters {
            parameters: self.parameters.clone().with_oscillator(
                OscillatorTypeDistribution::Pulse(Uniform::new(
                    duty_cycle_range.0,
                    duty_cycle_range.1,
                )),
                probability,
                amplitude_range,
            ),
        }
    }

    /// Adds a triangle oscillator with the given amplitude range.
    #[args(probability = "1.")]
    #[pyo3(text_signature = "(self, probability, amplitude_range, /)")]
    pub fn add_triangle(&self, probability: f64, amplitude_range: (f32, f32)) -> Self {
        DataParameters {
            parameters: self.parameters.clone().with_oscillator(
                OscillatorTypeDistribution::Triangle,
                probability,
                amplitude_range,
            ),
        }
    }

    /// Adds a noise oscillator with the given amplitude range.
    /// This will create samples that are normally distributed around `0.0` with the amplitude as standard deviation and any samples greater than the amplitude being cut off.
    #[args(probability = "1.")]
    #[pyo3(text_signature = "(self, probability, amplitude_range, /)")]
    pub fn add_noise(&self, probability: f64, amplitude_range: (f32, f32)) -> Self {
        DataParameters {
            parameters: self.parameters.clone().with_oscillator(
                OscillatorTypeDistribution::Noise,
                probability,
                amplitude_range,
            ),
        }
    }

    /// Adds a distortion effect to all samples.
    /// The `power_range` argument determines the strength of the distortion.
    #[args(probability = "1.", power_range = "(1., 1.)")]
    #[pyo3(text_signature = "(self, probability, power_range, /)")]
    pub fn apply_distortion(&self, probability: f64, power_range: (f32, f32)) -> Self {
        DataParameters {
            parameters: self
                .parameters
                .clone()
                .with_effect(EffectTypeDistribution::distortion(power_range), probability),
        }
    }

    #[pyo3(text_signature = "(self, /)")]
    pub fn apply_normalization(&self, probability: f64) -> Self {
        DataParameters {
            parameters: self
                .parameters
                .clone()
                .with_effect(EffectTypeDistribution::Normalize, probability),
        }
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

#[pyfunction]
#[pyo3(text_signature = "(path, /)")]
pub fn load_wav(path: &str) -> Result<Audio> {
    audio_samples::Audio::from_wav(path).map(|audio| audio.into())
}

#[pymethods]
impl Audio {
    /// Loads an audio clip from the given path.
    #[classmethod]
    #[pyo3(text_signature = "(path, /)")]
    fn from_wav(_cls: &PyType, path: &str) -> Result<Self> {
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

    /// The norm of the fourier transform of the audio.
    #[pyo3(text_signature = "(self, /)")]
    fn fft<'py>(&self, py: Python<'py>) -> &'py PyArray<f32, Dim<[usize; 1]>> {
        PyArray::from_vec(
            py,
            self.audio
                .fft()
                .into_iter()
                .map(|x| x.norm() / (self.audio.num_samples() as f32))
                .collect(),
        )
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

#[pyclass]
#[derive(Clone)]
pub struct DataPointLabel {
    label: audio_samples::data::DataPointLabel,
}

#[pymethods]
impl DataPointLabel {
    /// The fundamental frequency of the audio.
    #[pyo3(text_signature = "(self, /)")]
    fn frequency(&self) -> Option<f32> {
        self.label.base_frequency
    }

    /// The fundamental frequency of the audio mapped into the range `[-1;1]`.
    #[pyo3(text_signature = "(self, /)")]
    fn frequency_map(&self) -> Option<f32> {
        self.label.base_frequency_map
    }

    #[pyo3(text_signature = "(self, /)")]
    fn note_number(&self) -> Option<f32> {
        self.frequency()
            .map(|freq| audio_samples::frequency_to_note_number(freq))
    }

    fn chord_type(&self) -> u32 {
        self.label.chord_type
    }

    fn chord_type_name(&self) -> String {
        chord_type_name(self.chord_type())
    }
}

impl From<audio_samples::data::DataPointLabel> for DataPointLabel {
    fn from(label: audio_samples::data::DataPointLabel) -> Self {
        Self { label }
    }
}

/// Represents a data point with an audio clip and the parameters used to generate it.
#[pyclass]
#[derive(Clone)]
pub struct DataPoint {
    signal: Audio,
    label: DataPointLabel,
}

#[pymethods]
impl DataPoint {
    /// The audio.
    #[pyo3(text_signature = "(self, /)")]
    fn audio(&self) -> Audio {
        self.signal.clone()
    }

    /// A vector of the samples in the audio.
    #[pyo3(text_signature = "(self, /)")]
    fn samples<'py>(&self, py: Python<'py>) -> &'py PyArray<f32, Dim<[usize; 1]>> {
        self.signal.samples(py)
    }

    #[pyo3(text_signature = "(self, /)")]
    fn label(&self) -> DataPointLabel {
        self.label.clone()
    }

    /// The fundamental frequency of the audio.
    #[pyo3(text_signature = "(self, /)")]
    fn frequency(&self) -> Option<f32> {
        self.label.frequency()
    }

    /// The fundamental frequency of the audio mapped into the range `[-1;1]`.
    #[pyo3(text_signature = "(self, /)")]
    fn frequency_map(&self) -> Option<f32> {
        self.label.frequency_map()
    }

    /// Saves the audio to a wav file.
    #[pyo3(text_signature = "(self, path, /)")]
    fn audio_to_wav(&self, path: &str) -> Result<()> {
        self.signal.to_wav(path)
    }
}

impl From<data::DataPoint> for DataPoint {
    fn from(data_point: data::DataPoint) -> Self {
        Self {
            signal: data_point.audio.into(),
            label: DataPointLabel {
                label: data::DataPointLabel::new(&data_point.parameters),
            },
        }
    }
}

/// Represents a data point with an audio clip and the parameters used to generate it.
#[pyclass]
#[derive(Clone)]
pub struct DataSet {
    data: Vec<DataPoint>,
}

#[pymethods]
impl DataSet {
    fn __len__(&self) -> usize {
        self.data.len()
    }

    fn __getitem__(&self, index: usize) -> Option<DataPoint> {
        self.data.get(index).cloned()
    }

    fn random_partition(&self, p: f32) -> (Self, Self) {
        let mut rng = rand::thread_rng();
        let mut data = self.data.clone();
        data.shuffle(&mut rng);
        let n = (data.len() as f32 * p).round() as usize;
        let (a, b) = data.split_at(n);
        (Self { data: a.to_vec() }, Self { data: b.to_vec() })
    }
}

#[pyfunction]
#[pyo3(text_signature = "(path, /)")]
pub fn load_data_set(path: &str) -> Result<DataSet> {
    audio_samples::data::load_dir(path).map(|data| DataSet {
        data: data
            .into_iter()
            .map(|(audio, label)| DataPoint {
                signal: audio.into(),
                label: label.into(),
            })
            .collect(),
    })
}

/// A Python module implemented in Rust.
#[pymodule]
fn audio_samples_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Audio>()?;
    m.add_class::<DataPoint>()?;
    m.add_class::<DataParameters>()?;
    m.add_class::<DataPointLabel>()?;
    m.add_class::<DataSet>()?;
    m.add_function(wrap_pyfunction!(debug_txt, m)?)?;
    m.add_function(wrap_pyfunction!(cent_diff, m)?)?;
    m.add_function(wrap_pyfunction!(load_wav, m)?)?;
    m.add_function(wrap_pyfunction!(load_data_set, m)?)?;
    m.add_function(wrap_pyfunction!(map_to_note_number, m)?)?;
    m.add_function(wrap_pyfunction!(note_number_to_map, m)?)?;
    m.add_function(wrap_pyfunction!(frequency_to_map, m)?)?;
    m.add_function(wrap_pyfunction!(map_to_frequency, m)?)?;
    m.add_function(wrap_pyfunction!(num_chord_types, m)?)?;
    m.add_function(wrap_pyfunction!(chord_type_name, m)?)?;
    Ok(())
}
