import pandas as pd
import numpy as np
import torch
import time
import queue
import threading
import csv
from collections import deque
import os

class RealtimeFallDetector:
    def __init__(self, predictor, window_size=10, fall_threshold=3,
                 data_source=None, sampling_rate=0.1):
        """
        Initialize the real-time fall detector

        Args:
            predictor: The trained fall detection model predictor
            window_size: Size of the sliding window (number of frames/predictions)
            fall_threshold: Minimum number of positive fall predictions to trigger an alert
            data_source: Source of real-time data (None for real sensors)
            sampling_rate: Time between sensor readings in seconds
        """
        self.predictor = predictor
        self.window_size = window_size
        self.fall_threshold = fall_threshold
        self.data_source = data_source
        self.sampling_rate = sampling_rate

        # Sliding window for predictions (1: fall, 0: normal)
        self.prediction_window = deque(maxlen=window_size)

        # Data buffer for storing point cloud frames
        self.data_buffer = queue.Queue()

        # Real-time monitoring state
        self.is_running = False
        self.monitor_thread = None
        self.data_thread = None

        # For temporary data storage before saving to csv
        self.current_frame_data = []
        self.temp_csv_path = "temp_radar_data.csv"

    def start_monitoring(self):
        """Start the real-time monitoring threads"""
        if self.is_running:
            print("Monitoring is already running")
            return

        self.is_running = True

        # Start data collection thread
        self.data_thread = threading.Thread(target=self._collect_data)
        self.data_thread.daemon = True
        self.data_thread.start()

        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_falls)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

        print("Real-time fall detection started")

    def stop_monitoring(self):
        """Stop the real-time monitoring"""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        if self.data_thread:
            self.data_thread.join(timeout=1)
        print("Real-time fall detection stopped")

    def _collect_data(self):
        """Collect data from sensors or simulator"""
        if self.data_source:
            # Simulate data from file for testing
            self._simulate_data_from_source()
        else:
            # Collect data from real sensors
            self._collect_sensor_data()

    def _simulate_data_from_source(self):
        """Simulate real-time data from a file source"""
        try:
            df = pd.read_csv(self.data_source)
            for _, row in df.iterrows():
                if not self.is_running:
                    break

                # Put frame data into the buffer
                self.data_buffer.put(row)

                # Save to current frame for potential storage
                self.current_frame_data.append(row.to_dict())

                # Simulate real-time by waiting
                time.sleep(self.sampling_rate)

        except Exception as e:
            print(f"Error simulating data: {str(e)}")

    def _collect_sensor_data(self):
        """Collect real-time data from radar sensors"""
        # This function would interface with your actual radar hardware
        # For demonstration, this is a placeholder

        try:
            # Setup connection to radar device
            # radar = RadarSensor(port='COM3')  # Example - replace with actual implementation

            while self.is_running:
                # Read data from radar
                # point_cloud = radar.get_point_cloud()  # Example - replace with actual code

                # For demonstration, generating dummy data
                point_cloud = np.random.rand(np.random.randint(10, 50), 5)

                # Create a frame entry
                frame_data = {
                    'timestamp': time.time(),
                    'point_cloud': point_cloud.tolist()
                }

                # Add to buffer for processing
                self.data_buffer.put(frame_data)

                # Save to current frame for potential storage
                self.current_frame_data.append(frame_data)

                # If buffer gets too large, save to disk
                if len(self.current_frame_data) >= 100:
                    self._save_buffer_to_csv()

                # Wait for next sample
                time.sleep(self.sampling_rate)

        except Exception as e:
            print(f"Error collecting sensor data: {str(e)}")

    def _save_buffer_to_csv(self):
        """Save collected data to CSV file"""
        try:
            with open(self.temp_csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.current_frame_data[0].keys())

                # Write header if file is new
                if f.tell() == 0:
                    writer.writeheader()

                writer.writerows(self.current_frame_data)

            # Clear the buffer after saving
            self.current_frame_data = []

        except Exception as e:
            print(f"Error saving data to CSV: {str(e)}")

    def _monitor_falls(self):
        """Monitor the data buffer and detect falls"""
        while self.is_running:
            try:
                # Check if there's data to process
                if self.data_buffer.empty():
                    time.sleep(0.1)  # Small delay to prevent CPU hogging
                    continue

                # Get frame data
                frame_data = self.data_buffer.get()

                # Save frame to temporary csv for prediction
                with open(self.temp_csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp', 'point_cloud'])
                    writer.writerow([frame_data.get('timestamp', time.time()),
                                     str(frame_data.get('point_cloud', []))])

                # Make prediction
                result = self.predictor.predict(self.temp_csv_path)

                # Add prediction to sliding window
                prediction = 1 if result['prediction'] == 1 else 0
                self.prediction_window.append(prediction)

                # Check if fall detected based on window
                if self._is_fall_detected():
                    self._trigger_fall_alert(result['confidence'])

                # 展示每一段数据的检测结果和置信度
                print(f"Data Prediction: {'Fall' if prediction == 1 else 'Normal'}, Confidence: {result['confidence'] * 100:.2f}%")

            except Exception as e:
                print(f"Error in fall monitoring: {str(e)}")
                time.sleep(0.5)  # Delay before retry

    def _is_fall_detected(self):
        """Check if a fall is detected based on the sliding window"""
        if len(self.prediction_window) < self.window_size:
            return False

        # Count positive fall predictions in the window
        fall_count = sum(self.prediction_window)

        # Trigger if count exceeds threshold
        return fall_count >= self.fall_threshold

    def _trigger_fall_alert(self, confidence):
        """Trigger an alert when a fall is detected"""
        print("\n" + "!" * 50)
        print(f"FALL DETECTED! Confidence: {confidence * 100:.2f}%")
        print("!" * 50 + "\n")

        # Here you could add code to:
        # - Send SMS/email alerts
        # - Trigger alarm systems
        # - Log the event
        # - Capture and store video/data around the fall event

        # Example: Log to file
        with open("fall_alerts.log", "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Fall detected with {confidence * 100:.2f}% confidence\n")


class LongTermCSVProcessor:
    def __init__(self, predictor, csv_path, window_size=10, fall_threshold=3,
                 frame_interval=0.1, batch_size=40):
        """
        Initialize the long-term CSV data processor

        Args:
            predictor: The trained fall detection model predictor
            csv_path: Path to the large CSV file with long-term data
            window_size: Size of the sliding window (number of frames)
            fall_threshold: Minimum number of positive fall predictions to trigger a detection
            frame_interval: Time between consecutive frames in seconds
            batch_size: Number of frames to process at once (should match predictor's max_frames)
        """
        self.predictor = predictor
        self.csv_path = csv_path
        self.window_size = window_size
        self.fall_threshold = fall_threshold
        self.frame_interval = frame_interval
        self.batch_size = batch_size

        # For storing detected falls
        self.detected_falls = []

    def process_file(self, output_file=None):
        """Process the entire CSV file and detect falls"""
        try:
            # Read the CSV file
            df = pd.read_csv(self.csv_path)
            total_frames = len(df)

            print(f"Processing {total_frames} frames from {self.csv_path}")

            # Sliding window for predictions
            prediction_window = deque(maxlen=self.window_size)

            # Process the file in batches
            for start_idx in range(0, total_frames, 10):  # 50% overlap between batches
                end_idx = min(start_idx + self.batch_size, total_frames)

                # Extract batch
                batch_df = df.iloc[start_idx:end_idx].copy()

                # Save batch to temporary file for prediction
                temp_file = f"./tmp/temp_batch_{start_idx}.csv"
                batch_df.to_csv(temp_file, index=False)

                # Make prediction on the batch
                result = self.predictor.predict(temp_file)

                # Add prediction to window (仅当置信度 >= 98% 时判定为跌倒)
                prediction = 1 if result['confidence'] >= 0.98 else 0
                prediction_window.append(prediction)

                # 展示每一段数据的检测结果和置信度
                print(f"start_idx: {start_idx}  Data Prediction: {'Fall' if prediction == 1 else 'Normal'}, Confidence: {result['confidence'] * 100:.2f}%")

                # Check if fall detected
                timestamp = batch_df.iloc[0].get('timestamp', start_idx * self.frame_interval)
                if len(prediction_window) == self.window_size and sum(prediction_window) >= self.fall_threshold:
                    fall_info = {
                        'start_time': timestamp,
                        'end_time': timestamp + (self.batch_size * self.frame_interval),
                        'confidence': result['confidence'],
                        'frame_index': start_idx
                    }

                    self.detected_falls.append(fall_info)

                    print(
                        f"Fall detected at frame {start_idx} (time: {timestamp:.2f}s) with confidence {result['confidence'] * 100:.2f}%")

                    prediction_window.clear()  # 新增清空队列操作

                # Clean up temporary file
                if os.path.exists(temp_file):
                    os.remove(temp_file)

                # Progress update
                if start_idx % 100 == 0:
                    print(f"Processed {start_idx}/{total_frames} frames ({start_idx / total_frames * 100:.1f}%)")

            # 生成输出文件名，前缀和输入文件名一样
            input_filename = os.path.basename(self.csv_path)
            input_prefix = os.path.splitext(input_filename)[0]
            output_file = f"{input_prefix}_detected_falls.csv"

            # Save results if output file specified
            if output_file and self.detected_falls:
                self._save_results(output_file)

            return self.detected_falls

        except Exception as e:
            print(f"Error processing CSV file: {str(e)}")
            return []

    def _save_results(self, output_file):
        """Save detected falls to a file"""
        try:
            # Convert to DataFrame and save
            result_df = pd.DataFrame(self.detected_falls)
            result_df.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")
        except Exception as e:
            print(f"Error saving results: {str(e)}")


# Example usage
if __name__ == "__main__":
    from predict import Predictor

    # Initialize the predictor
    predictor = Predictor(
        model_path="../best_model/111/best_model_epoch7_20250413_235216.pth",
        model_type="HybridModel",
        max_frames=40,
        max_points=100,
        method='mask'
    )

    # Real-time detection example
    print("Starting real-time fall detection...")
    data_source = "F:\\code\\rada\\script\\origin_data_to_csv\\serial_data\\pointCloud_20250414_18-16-21.264_xlm_fallSit_count2.csv"
    # rt_detector = RealtimeFallDetector(
    #     predictor=predictor,
    #     window_size=5,  # Consider 5 predictions in the window
    #     fall_threshold=3,  # Trigger if 3 or more are positive
    #     data_source=data_source,  # For testing with simulated data
    #     sampling_rate=0.2  # 5 samples per second
    # )
    #
    # # Start monitoring (runs in background threads)
    # rt_detector.start_monitoring()

    # try:
    #     # Keep main thread alive for demo
    #     print("Press Ctrl+C to stop...")
    #     while True:
    #         time.sleep(1)
    # except KeyboardInterrupt:
    #     print("Stopping monitoring...")
    #     rt_detector.stop_monitoring()

    # Process long-term CSV file
    print("\nProcessing long-term CSV file...")
    csv_processor = LongTermCSVProcessor(
        predictor=predictor,
        csv_path=data_source,
        window_size=10,  # Consider 8 consecutive predictions
        fall_threshold=7,  # Trigger if 4 or more are positive
        frame_interval=0.1,  # 0.1 seconds between frames
        batch_size=40  # Process 40 frames at once
    )

    # Process the file
    detected_falls = csv_processor.process_file()

    print(f"Found {len(detected_falls)} falls in the long-term data")