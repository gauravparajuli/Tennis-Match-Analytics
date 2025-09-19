from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker

def main():
    print('main is running')
    input_video_path = r'input_vid/input_video.mp4'
    frames = read_video(input_video_path)

    # detect players
    player_tracker = PlayerTracker(model_path='yolov8x')
    ball_tracker = BallTracker(model_path=r'weights\ball_detection\best.pt')
    player_detections = player_tracker.detect_frames(frames, read_from_stub=True, stub_path='tracker_stubs/player_detections.pkl')
    ball_detections = ball_tracker.detect_frames(frames, read_from_stub=True, stub_path='tracker_stubs/ball_detections.pkl')

    # draw output
    ## draw player bounding boxes
    output_video_frames = player_tracker.draw_bboxes(frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(frames, ball_detections)

    # save video
    save_video(output_video_frames, 'output_vid/output.avi')

if __name__ == "__main__":
    main()