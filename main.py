import sounddevice as sd
import numpy as np
from scipy.signal import butter, lfilter
import sys
import argparse
from typing import Optional
import matplotlib.pyplot as plt
from scipy.io.wavfile import write as write_wav, read as read_wav  # Import the WAV writing function

# Parameters
RATE: int = 96000  # Sample rate
CHUNK: int = 1024  # Buffer size
BIT_DURATION: float = 0.1  # Default bit duration in seconds

# Create a global variable to store the output stream
output_stream: Optional[sd.OutputStream] = None


def generate_sine_wave(frequency: float, duration: float, sample_rate: int) -> np.ndarray:
    t: np.ndarray = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return 0.5 * np.sin(2 * np.pi * frequency * t)  # Amplitude scaled to 0.5

def bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: int, order: int = 5) -> np.ndarray:
    nyquist: float = 0.5 * fs
    low: float = lowcut / nyquist
    high: float = highcut / nyquist
    b, a = butter(order, [low, high], btype='bandpass')
    return lfilter(b, a, data)

# Function to design a bandstop filter
def bandstop_filter(data: np.ndarray, lowcut: float, highcut: float, fs: int, order: int = 5) -> np.ndarray:
    nyquist: float = 0.5 * fs
    low: float = lowcut / nyquist
    high: float = highcut / nyquist
    b, a = butter(order, [low, high], btype='bandstop')
    return lfilter(b, a, data)

def decode_frequency(
    fft_data: np.ndarray,
    sample_rate: int,
    low_freq: float,
    high_freq: float,
    threshold: float
) -> str:
    """Identifies which frequency bands are present (1) or absent (0)"""
    positive_freqs: np.ndarray = np.fft.fftfreq(len(fft_data), d=1/sample_rate)[:len(fft_data)//2]
    positive_magnitudes: np.ndarray = np.abs(fft_data[:len(fft_data)//2])

    # Calculate bands for 8 bits (one byte)
    freq_bands: np.ndarray = np.linspace(low_freq, high_freq, 9)  # 9 points for 8 bands
    
    # For each band, check if there's significant frequency content
    bits: list[str] = []
    
    for i in range(8):  # 8 bits
        band_mask: np.ndarray = (positive_freqs >= freq_bands[i]) & (positive_freqs < freq_bands[i + 1])
        band_power: float = np.sum(positive_magnitudes[band_mask])
        bits.append('1' if band_power > threshold else '0')
    
    return ''.join(bits)

def process_audio_block(block: list[int], out_fp: Optional[any], print_stdout: bool = False) -> None:
    """Process a block of decoded bytes, handle error correction and output"""
    if len(block) < 2:  # Need at least one data byte and parity
        return

    block_data: list[int] = block[:-1]
    received_parity: int = block[-1]
    # Only calculate parity on actual data bytes
    calculated_parity: int = calculate_parity(block_data)

    if calculated_parity != received_parity:
        print(f"Error detected in block! Expected: {received_parity}, Got: {calculated_parity}")
        # Try to correct the error
        for j in range(len(block_data)):
            for bit_index in range(8):
                original_byte: int = block_data[j]
                block_data[j] ^= (1 << bit_index)
                new_parity: int = calculate_parity(block_data)

                if new_parity == received_parity:
                    print(f"Error corrected: Byte {j}, Bit {bit_index}")
                    break

                block_data[j] = original_byte
        else:
            print("Uncorrectable error")

    # Output all bytes in the block
    for byte in block_data:
        if print_stdout:
            print(chr(byte), end='', flush=True)
        if out_fp:
            out_fp.write(bytes([byte]))

def plot_fft_data(fft_data: np.ndarray, sample_rate: int, threshold: float = None) -> None:
    """Plot FFT data in real-time"""
    plt.clf()  # Clear the current figure
    freqs = np.fft.fftfreq(len(fft_data), d=1/sample_rate)
    magnitudes = np.abs(fft_data)
    plt.plot(freqs[:len(freqs)//2], magnitudes[:len(magnitudes)//2])
    if threshold is not None:
        plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    plt.title("Real-time FFT")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.pause(0.01)  # Small pause to allow plot to update

def decode_audio_data(
    audio_data: np.ndarray,
    sample_rate: int,
    bit_duration: float,
    out_fp: Optional[any] = None,
    print_stdout: bool = False,
    plot_fft: bool = False
) -> None:
    """Decode audio data into bytes using frequency analysis"""
    if plot_fft:
        plt.ion()  # Enable interactive plotting
        plt.figure(figsize=(10, 4))

    samples_per_bit: int = int(sample_rate * bit_duration)
    block_size: int = 9  # 8 data bytes + 1 parity byte
    block: list[int] = []
    block_count: int = 0
    
    # Initialize moving average and alpha
    alpha: float = 0.2  # Smoothing factor for moving average
    moving_average: float = 0.0

    # Process the audio data in chunks
    for i in range(0, len(audio_data), samples_per_bit):
        chunk: np.ndarray = audio_data[i:i + samples_per_bit]
        if len(chunk) < samples_per_bit:
            chunk = np.pad(chunk, (0, samples_per_bit - len(chunk)), 'constant')

        fft_data: np.ndarray = np.fft.fft(chunk)
        nyquist: float = sample_rate / 2

        positive_magnitudes: np.ndarray = np.abs(fft_data[:len(fft_data)//2])
        current_max: float = np.max(positive_magnitudes)

        moving_average = alpha * current_max + (1 - alpha) * moving_average
        threshold: float = moving_average * 0.7

        lower_bound: float = 1000
        upper_bound: float = nyquist - 1000

        bits: str = decode_frequency(fft_data, sample_rate, lower_bound, upper_bound, threshold)
        if bits and len(bits) == 8:
            char_code: int = int(bits, 2)
            block.append(char_code)

            if len(block) == block_size:
                block_count += 1
                print(f"Processing block {block_count}: {len(block)-1} data bytes")
                process_audio_block(block, out_fp, print_stdout)
                block = []

        if plot_fft:
            plot_fft_data(fft_data, sample_rate, threshold)

    # Process any remaining bytes in the last block
    if block:
        block_count += 1
        print(f"Processing final block {block_count}: {len(block)-1} data bytes")
        process_audio_block(block, out_fp, print_stdout)

    if plot_fft:
        plt.ioff()  # Disable interactive plotting
        plt.show()  # Keep the final plot window open

def audio_callback(
    indata: np.ndarray,
    frames: int,
    time,
    status: sd.CallbackFlags,
    output_file: Optional[str] = None,
    print_stdout: bool = False,
    plot_fft: bool = False
) -> None:
    if status:
        print(f"Status: {status}", flush=True)
    
    out_fp = open(output_file, "ab") if output_file else None
    try:
        decode_audio_data(indata[:, 0], RATE, BIT_DURATION, out_fp, print_stdout, plot_fft)
    finally:
        if out_fp:
            out_fp.close()


def encode_file_to_frequencies(file_path: str, low_freq: float, high_freq: float, plot_fft: bool = False) -> np.ndarray:
    """Encodes a file by representing each bit with presence/absence of a frequency band."""
    with open(file_path, "rb") as f:
        file_data: bytes = f.read()

    # Calculate 8 frequency bands (one per bit)
    freq_bands: np.ndarray = np.linspace(low_freq, high_freq, 9)  # 9 points for 8 bands
    print(f"Using 8 frequency bands, encoding 1 byte per symbol")

    # Generate sine waves for each byte
    sine_waves: list[np.ndarray] = []
    block_size: int = 8  # 8 data bytes per block
    
    # Process input bytes in blocks
    for i in range(0, len(file_data), block_size):
        # Get current block of data
        block: bytes = file_data[i:i + block_size]
        print(f"Encoding block {i//block_size + 1}: {len(block)} bytes")
        
        # Calculate parity for this block
        parity: int = calculate_parity(list(block))
        
        # Encode each byte in the block
        for byte in block:
            bits: str = format(byte, '08b')
            t: np.ndarray = np.linspace(0, BIT_DURATION, int(RATE * BIT_DURATION), endpoint=False)
            composite: np.ndarray = np.zeros_like(t)
            
            for j, bit in enumerate(bits):
                if bit == '1':
                    freq: float = (freq_bands[j] + freq_bands[j + 1]) / 2
                    composite += 0.5 * np.sin(2 * np.pi * freq * t)
            
            sine_waves.append(composite)
        
        # Encode the parity byte
        parity_bits: str = format(parity, '08b')
        t: np.ndarray = np.linspace(0, BIT_DURATION, int(RATE * BIT_DURATION), endpoint=False)
        composite: np.ndarray = np.zeros_like(t)
        
        for j, bit in enumerate(parity_bits):
            if bit == '1':
                freq: float = (freq_bands[j] + freq_bands[j + 1]) / 2
                composite += 0.5 * np.sin(2 * np.pi * freq * t)
        
        sine_waves.append(composite)
    
    return np.concatenate(sine_waves)

# Function to calculate parity byte
def calculate_parity(data: list[int]) -> int:
    parity: int = 0
    for byte in data:
        parity ^= byte
    return parity


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Sound transfer utility")
        parser.add_argument("-r", type=int, default=RATE, help="Set the sample rate (default: %(default)s Hz)")
        parser.add_argument("-c", type=int, default=CHUNK, help="Set the chunk size (default: %(default)s)")
        parser.add_argument("-b", type=float, default=BIT_DURATION, help="Set the bit duration in seconds (default: %(default)s)")
        parser.add_argument("-fi", type=str, help="Path to the input file")
        parser.add_argument("-fo", type=str, help="Path to the output file")
        parser.add_argument("-m", type=str, choices=['encode', 'decode'], default='decode', help="Mode: encode or decode (default: %(default)s)")
        parser.add_argument("--plot", action="store_true", help="Visualize the Fourier decomposition of the encoded signal")
        parser.add_argument("--stdout", action="store_true", help="Print decoded output to stdout")
        parser.add_argument("--play", action="store_true", help="Play the encoded signal")
        args = parser.parse_args()

        RATE = args.r
        CHUNK = args.c
        BIT_DURATION = args.b

        nyquist: float = 0.5 * RATE

        if args.m == 'encode':
            if not args.fi:
                print("Error: Input file is required in encode mode.")
                sys.exit(1)

            file_path: str = args.fi
            print(f"Encoding file: {file_path}")

            # Encode the file into frequencies
            encoded_signal: np.ndarray = encode_file_to_frequencies(file_path, low_freq=0, high_freq=nyquist, plot_fft=args.plot)
            print("file encoded.")

            output_file: Optional[str] = args.fo
            if output_file:
                print(f"Saving encoded signal to: {output_file}")
                scaled_signal: np.ndarray = np.int16(encoded_signal / np.max(np.abs(encoded_signal)) * 32767)
                write_wav(output_file, RATE, scaled_signal)

            if args.play:  # Only play if --play flag is set
                print("playing...")
                sd.play(encoded_signal, samplerate=RATE)
                sd.wait()

            if args.plot:
                plt.figure(figsize=(10, 4))
                # Plot the FFT of the entire encoded signal
                fft_data = np.fft.fft(encoded_signal)
                plot_fft_data(fft_data, RATE)
                plt.ioff()
                plt.show()
        elif args.m == 'decode':
            print("Decoding mode...")
            if args.fi:
                try:
                    RATE, indata = read_wav(args.fi)
                    if len(indata.shape) > 1:
                        indata = indata[:, 0]
                    
                    out_fp = open(args.fo, 'wb') if args.fo else None
                    try:
                        decode_audio_data(indata, RATE, BIT_DURATION, out_fp, args.stdout, args.plot)
                    finally:
                        if out_fp:
                            out_fp.close()
                except Exception as e:
                    print(f"Error reading or decoding WAV file: {e}")
                    sys.exit(1)
            else:
                # Microphone input
                output_stream = sd.OutputStream(samplerate=RATE, blocksize=CHUNK, channels=1)
                output_stream.start()
                with sd.InputStream(
                    samplerate=RATE,
                    blocksize=CHUNK,
                    channels=1,
                    callback=lambda indata, frames, time, status: audio_callback(
                        indata, frames, time, status, args.fo, args.stdout, args.plot
                    )
                ):
                    sd.sleep(int(1e6))
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        if output_stream:
            output_stream.stop()
            output_stream.close()

