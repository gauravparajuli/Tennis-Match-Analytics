from utils import read_video, save_video

def main():
    print('main is running')
    input_video_path = r'input_vid/input_video.mp4'
    frames = read_video(input_video_path)

    # save video
    save_video(frames, 'output_vid/output.avi')

if __name__ == "__main__":
    main()