# Reference Dance Videos (Motion Transfer)

Place reference dance videos here for pose extraction and dance-style generation.

- **rat_dance.mp4** — RAT Dance Challenge. Used when `motion_id=rat_dance`.

The backend uses MediaPipe Pose to extract body keypoints from each frame, then normalizes the motion. Generated videos use LTX-2 image-to-video with a dance prompt; pose conditioning (AnimateDiff + ControlNet) can be added later.
