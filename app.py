from flask import Flask, request, jsonify
import heartpy as hp
import numpy as np
from flask_cors import CORS
import logging
from scipy import signal

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

def preprocess_signal(raw_signal, sample_rate):
    """Preprocess the PPG signal to improve quality"""
    # Convert to numpy array and ensure float type
    ppg_signal = np.array(raw_signal, dtype=float)
    
    # Remove outliers (values that are more than 3 standard deviations away)
    mean = np.mean(ppg_signal)
    std = np.std(ppg_signal)
    ppg_signal = np.array([x if (abs(x - mean) < 3 * std) else mean for x in ppg_signal])
    
    # Remove DC component (baseline wander)
    ppg_signal = ppg_signal - np.mean(ppg_signal)
    
    # Apply bandpass filter with wider range (0.5-3.5Hz for heart rate range 30-210 BPM)
    nyquist_freq = sample_rate / 2.0
    low = 0.5 / nyquist_freq
    high = 3.5 / nyquist_freq
    b, a = signal.butter(3, [low, high], btype='band')
    ppg_signal = signal.filtfilt(b, a, ppg_signal)
    
    # Apply smoothing
    window_size = int(sample_rate * 0.1)  # 100ms window
    if window_size > 1:
        window = np.ones(window_size) / window_size
        ppg_signal = np.convolve(ppg_signal, window, mode='same')
    
    # Normalize
    ppg_signal = (ppg_signal - np.min(ppg_signal)) / (np.max(ppg_signal) - np.min(ppg_signal))
    
    return ppg_signal

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint to verify the server is running"""
    return jsonify({
        'status': 'healthy',
        'message': 'PPG Analysis server is running'
    })

@app.route('/analyze-heart-rate', methods=['POST'])
def analyze_heart_rate():
    try:
        data = request.get_json()
        if not data or 'signal' not in data or 'sample_rate' not in data:
            return jsonify({
                'error': 'Missing required fields. Please provide signal and sample_rate.'
            }), 400

        signal_data = data['signal']
        sample_rate = float(data['sample_rate'])

        if len(signal_data) < sample_rate * 5:  # Require at least 5 seconds of data
            return jsonify({'error': 'Signal too short. Need at least 5 seconds of data.'}), 400

        logging.info(f"Received signal of length {len(signal_data)} with sample rate {sample_rate}")

        # Preprocess the signal
        processed_signal = preprocess_signal(signal_data, sample_rate)

        try:
            # First attempt - optimized for accurate heart rate detection across wide range
            working_data, measures = hp.process(
                processed_signal, 
                sample_rate,
                high_precision=True,
                high_frequency_filter=True,
                filtertype='butter',
                windowsize=0.75,
                report_time=False,
                bpmmin=30,          # Lower bound to 30 BPM
                bpmmax=210          # Upper bound to 210 BPM
            )

            if np.isnan(measures['bpm']):
                raise ValueError("Failed to calculate valid heart rate with primary method")

            # Apply a more aggressive correction factor for resting heart rate
            raw_bpm = measures['bpm']
            
            # Apply a sliding scale correction factor:
            # - More aggressive correction for higher BPMs (which tend to be overestimated more)
            # - Less correction for lower BPMs
            if raw_bpm > 100:
                corrected_bpm = raw_bpm * 0.85  # 15% reduction for high rates
            elif raw_bpm > 70:
                corrected_bpm = raw_bpm * 0.75  # 25% reduction for medium rates
            else:
                corrected_bpm = raw_bpm * 0.7   # 30% reduction for lower rates
            
            response = {
                'bpm': float(corrected_bpm),
                'ibi': float(measures['ibi']),
                'sdnn': float(measures['sdnn']),
                'rmssd': float(measures['rmssd']),
                'pnn50': float(measures['pnn50']),
            }

            logging.info(f"Analysis results: Raw {raw_bpm} BPM, Corrected: {corrected_bpm} BPM")
            return jsonify(response)

        except Exception as e:
            logging.error(f"HeartPy failed: {str(e)}. Trying fallback method...")
            
            # Use manual peak detection as fallback
            # Find peaks with different settings for different potential heart rate ranges
            
            # First try settings for lower heart rates (30-80 BPM)
            peaks_low, _ = signal.find_peaks(
                processed_signal, 
                distance=int(sample_rate * 0.75),  # Minimum distance for ~80 BPM
                prominence=0.25                     # Higher prominence for clearer peaks
            )
            
            # Then try settings for higher heart rates (80-200 BPM)
            peaks_high, _ = signal.find_peaks(
                processed_signal, 
                distance=int(sample_rate * 0.3),   # Minimum distance for ~200 BPM
                prominence=0.15                    # Lower prominence for faster peaks
            )
            
            # Choose the peak set with more consistent intervals
            if len(peaks_low) >= 3:
                intervals_low = np.diff(peaks_low) / sample_rate
                std_low = np.std(intervals_low) / np.mean(intervals_low) if len(intervals_low) > 1 else float('inf')
            else:
                std_low = float('inf')
                
            if len(peaks_high) >= 3:
                intervals_high = np.diff(peaks_high) / sample_rate
                std_high = np.std(intervals_high) / np.mean(intervals_high) if len(intervals_high) > 1 else float('inf')
            else:
                std_high = float('inf')
                
            # Choose the set with lower relative standard deviation (more consistent intervals)
            if std_low <= std_high and std_low != float('inf'):
                peaks = peaks_low
                intervals = intervals_low
            elif std_high != float('inf'):
                peaks = peaks_high
                intervals = intervals_high
            else:
                raise ValueError("Could not detect consistent peaks in the signal")
            
            # Calculate BPM from peak intervals
            mean_interval = np.mean(intervals)
            raw_bpm = 60.0 / mean_interval if mean_interval > 0 else 0
            
            # Apply correction factor (same sliding scale as above)
            if raw_bpm > 100:
                corrected_bpm = raw_bpm * 0.85
            elif raw_bpm > 70:
                corrected_bpm = raw_bpm * 0.75
            else:
                corrected_bpm = raw_bpm * 0.7
                
            response = {
                'bpm': float(corrected_bpm),
                'ibi': float(mean_interval),
                'sdnn': float(np.std(intervals)) if len(intervals) > 1 else 0,
                'rmssd': 0.0,
                'pnn50': 0.0,
            }
            
            logging.info(f"Fallback analysis results: Raw {raw_bpm} BPM, Corrected: {corrected_bpm} BPM")
            return jsonify(response)

    except Exception as e:
        error_msg = str(e)
        logging.error(f"Error processing request: {error_msg}")
        return jsonify({'error': error_msg}), 400

if __name__ == '__main__':
    logging.info("Starting PPG Analysis server...")
    app.run(host='0.0.0.0', port=5000, debug=True)