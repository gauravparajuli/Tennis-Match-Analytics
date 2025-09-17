from utils import read_video, save_video
from trackers import PlayerTracker

def main():
    print('main is running')
    input_video_path = r'input_vid/input_video.mp4'
    frames = read_video(input_video_path)

    # detect players
    player_tracker = PlayerTracker(model_path='yolov8x')
    player_detections = player_tracker.detect_frames(frames)

    # draw output
    ## draw player bounding boxes
    output_video_frames = player_tracker.draw_bboxes(frames, player_detections)


    # save video
    save_video(output_video_frames, 'output_vid/output.avi')

if __name__ == "__main__":
    main()