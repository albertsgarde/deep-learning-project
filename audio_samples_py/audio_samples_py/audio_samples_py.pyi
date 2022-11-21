def frequency_to_map(self, frequency: float) -> str:
    """
    Given a frequency, returns the corresponding frequency mapping.
    """

def map_to_frequency(self, map: float) -> str:
    """
    Given a frequency map value, returns the corresponding frequency.
    """

class DataParameters(object):
    """
    Create a new DataParameters object with no oscillators or effects.
    All data points generated will have frequency randomly distributed between min_frequency and max_frequency.

    :param num_samples: The number of samples in each data point generated.
    :param sample_rate: The sample rate of all audio generated. Defaults to 44100 and should generally be left here.
    :param min_frequency: The minimum frequency of any oscillator in the data set.
    :param max_frequency: The maximum frequency of any oscillator in the data set.
    :param possible_chord_types: Which chord types can appear.
    """
    def __init__(self, num_samples: int, octave_parameters, sample_rate: int = 44100, min_frequency: float = 20, max_frequency: float = 20000, min_frequency_std_dev: float = 0.5, max_frequency_std_dev: float = 3., possible_chord_types: list[int] = [0]) -> None: ...

    def with_seed_offset(self, seed_offset: int) -> 'DataParameters':
        """
        Creates a new DataParameters object with the same parameters as this one, but with a different seed offset.
        Two DataParameter objects with different seed offsets will produce independent data points.

        :param seed_offset: The new seed offset to use.
        """

    def add_sine(self, amplitude_range: tuple[float, float]) -> str:
        """
        Adds a sine wave with a random amplitude in the given range.

        Total maximum of all waves must be no more than 1.
        """

    def add_saw(self, amplitude_range: tuple[float, float]) -> str:
        """
        Adds a saw wave with a random amplitude in the given range.

        Total maximum of all waves must be no more than 1.
        """

    def add_pulse(self, amplitude_range: tuple[float, float], duty_cycle_range: tuple[float, float]) -> str:
        """
        Adds a pulse wave with random amplitude and duty cycle in the given ranges.
        The duty cycle is the ratio of the pulse width to the period.

        Total maximum of all waves must be no more than 1.
        """

    def add_triangle(self, amplitude_range: tuple[float, float]) -> str:
        """
        Adds a triangle wave with a random amplitude in the given range.

        Total maximum of all waves must be no more than 1.
        """

    def add_noise(self, amplitude_range: tuple[float, float]) -> str:
        """
        Adds a noise oscillator with the given amplitude range.
        This will create samples that are normally distributed around `0.0` with the amplitude as standard deviation and any samples greater than the amplitude being cut off.

        Total maximum of all waves must be no more than 1.
        """

    def apply_distortion(self, power_range: tuple[float, float]) -> str:
        """
        Adds a distortion effect to all samples.
        The `power_range` argument determines the strength of the distortion.
        """

    def generate_at_index(self, index: int) -> str:
        """
        Generates a samples at the given index.
        Calling this function multiple times with the same index will return the same samples.
        Calling this function multiple times with different indices will return (pseudo-)independent samples.
        """