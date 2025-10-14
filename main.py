from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtlineDetector
from mini_court import MiniCourt
import cv2

def main():
    print('main is running')
    input_video_path = r'input_vid/input_video.mp4'
    frames = read_video(input_video_path)

    # detect players and ball
    player_tracker = PlayerTracker(model_path='yolov8x')
    ball_tracker = BallTracker(model_path=r'weights\ball_detection\best.pt')
    player_detections = player_tracker.detect_frames(frames, read_from_stub=True, stub_path='tracker_stubs/player_detections.pkl')
    ball_detections = ball_tracker.detect_frames(frames, read_from_stub=True, stub_path='tracker_stubs/ball_detections.pkl')
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections) # interpolate the ball position in missing frame

    # detect court lines
    court_model_path = r'weights\tennis_court_keypoints.pth'
    court_line_detector = CourtlineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(frames[0])

    # choose persons that are nearest to the court as players
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    # Minicourt
    mini_court = MiniCourt(frames[0])

    # detect in which frames the ball was shot
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)
    print(ball_shot_frames)

    # convert positions to mini court positions
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_positions(player_detections, ball_detections, court_keypoints)

    # draw output
    ## draw player and ball bounding boxes
    output_video_frames = player_tracker.draw_bboxes(frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(frames, ball_detections)

    ## draw court keypoints
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    # Draw mini court
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, player_mini_court_detections)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, ball_mini_court_detections, color=(0,255,255))

    # draw frame number on top left corner
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f'Frame: {i}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # save video
    save_video(output_video_frames, 'output_vid/output.avi')

if __name__ == "__main__":
    main()