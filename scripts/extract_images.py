import shutil
import fire
import cv2

from pathlib import Path
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
from rosbags.image import message_to_cvimage


def main(rosbag_path: str,
         topic: str,
         output_dir: str,
         extension: str,
         save_every_n: int = 6):
    rosbag_path = Path(rosbag_path)
    output_dir = Path(output_dir)
    
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Opening ROS bag: {rosbag_path}")
    print(f"Extracting images from topic: {topic}")
    print(f"Output directory: {output_dir}")
    
    # Open the ROS bag
    with Reader(rosbag_path) as reader:
        # Get connections for the specified topic
        connections = [x for x in reader.connections if x.topic == topic]
        if not connections:
            print(f"Error: Topic '{topic}' not found in the bag file")
            available_topics = [conn.topic for conn in reader.connections]
            print(f"Available topics: {available_topics}")
            return
        
        print(f"Found {len(connections)} connection(s) for topic '{topic}'")
        
        # Read messages from the topic
        messages = reader.messages(connections=connections)
        
        frame_count = 0
        save_count = 1
        for connection, timestamp, rawdata in messages:
            try:
                # Deserialize the message
                msg = deserialize_cdr(rawdata, connection.msgtype)
                
                # Convert ROS image message to OpenCV image
                cv_image = message_to_cvimage(msg)
                
                save_count -= 1
                if save_count != 0:
                    continue
                save_count = save_every_n

                # Generate filename with leading zeros (6 digits)
                filename = f"{frame_count:06d}.{extension}"
                filepath = output_dir / filename
                
                # Save the image
                success = cv2.imwrite(str(filepath), cv_image)
                if not success:
                    print(f"Warning: Failed to save image {filename}")
                
                frame_count += 1
                
                # Print progress every 100 frames
                if frame_count % 100 == 0:
                    print(f"Processed {frame_count} frames...")
                    
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                continue
        
        print(f"Successfully extracted {frame_count} frames to {output_dir}")


if __name__ == "__main__":
    fire.Fire(main)
