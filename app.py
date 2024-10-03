from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import onnxruntime as ort
from moviepy.editor import VideoFileClip
# Khởi tạo ứng dụng Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MODEL_PATH'] = 'model/Hayao_64.onnx'


# Tải mô hình AnimeGAN
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']  # Đặt ưu tiên sử dụng GPU trước
ort_session = ort.InferenceSession(app.config['MODEL_PATH'], providers=providers)

# Kiểm tra nếu mô hình sử dụng GPU
print("Mô hình đang chạy trên:", ort_session.get_providers())

def convert_to_anime(image):
    # Đọc ảnh và lưu lại kích thước ban đầu
    original_size = (image.shape[1], image.shape[0])  # Lưu kích thước ban đầu (width, height)
    
    # Chuyển đổi ảnh sang định dạng RGB và resize về kích thước (256x256)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    
    # Chuẩn hóa ảnh
    image_resized = image_resized.astype(np.float32) / 127.5 - 1.0
    image_resized = np.expand_dims(image_resized, axis=0)
    
    # Chuyển đổi ảnh sang phong cách anime
    anime_image = ort_session.run(None, {ort_session.get_inputs()[0].name: image_resized})[0]
    anime_image = (anime_image + 1) * 127.5
    anime_image = np.clip(anime_image, 0, 255).astype(np.uint8)
    
    # Bỏ batch dimension
    anime_image = anime_image[0]
    
    # Resize ảnh đã chuyển đổi về kích thước ban đầu
    anime_image_resized = cv2.resize(anime_image, original_size, interpolation=cv2.INTER_AREA)
    
    return anime_image_resized

# Chuyển đổi video sang phong cách anime, giữ nguyên âm thanh
def convert_video_to_anime(video_path):
    output_video_path = video_path.rsplit('.', 1)[0] + '_anime.mp4'

    # Mở video bằng MoviePy
    with VideoFileClip(video_path) as video:
        # Lấy âm thanh từ video gốc
        audio = video.audio
        fps = video.fps 
        # Chuyển đổi từng khung hình
        def process_frame(frame):
            anime_frame = convert_to_anime(frame)
            bgr_frame = cv2.cvtColor(anime_frame, cv2.COLOR_RGB2BGR)
            return bgr_frame

        # Áp dụng chuyển đổi lên tất cả các khung hình
        anime_video = video.fl_image(process_frame)

        # Gắn lại âm thanh vào video đã chuyển đổi
        anime_video = anime_video.set_audio(audio).set_fps(fps)
        
        # Xuất video đã chuyển đổi
        anime_video.write_videofile(output_video_path, codec='libx264', audio_codec='aac', fps=fps)

    return output_video_path

# Trang chủ
@app.route('/')
def index():
    return render_template('index.html')

# Trang chuyển đổi ảnh
# Danh sách các định dạng ảnh được chấp nhận
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Hàm kiểm tra xem tệp tin có phải là ảnh không
def allowed_image(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

# Trang chuyển đổi ảnh
@app.route('/convert', methods=['GET', 'POST'])
def convert():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('convert.html', error='Không tìm thấy tệp tải lên.')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('convert.html', error='Chưa chọn tệp để tải lên.')
        
        # Kiểm tra nếu tệp là ảnh dựa trên định dạng
        if file and allowed_image(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Chuyển đổi ảnh sang phong cách anime
            image = cv2.imread(filepath)
            anime_image = convert_to_anime(image)
            anime_filename = 'anime_' + filename
            anime_filepath = os.path.join(app.config['UPLOAD_FOLDER'], anime_filename)
            
            # Lưu ảnh đã chuyển đổi
            cv2.imwrite(anime_filepath, cv2.cvtColor(anime_image, cv2.COLOR_RGB2BGR))
            
            return render_template('convert.html', original=filename, anime=anime_filename)
        else:
            return render_template('convert.html', error='Tệp tải lên không phải là định dạng ảnh hợp lệ.')
    
    return render_template('convert.html')

# Trang chuyển đổi video
ALLOWED_VIDEO_EXTENSIONS = {'mp4'}

# Hàm kiểm tra xem tệp tin có phải là video không
def allowed_video(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

@app.route('/convert2vid', methods=['GET', 'POST'])
def convert2vid():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('cvideo.html', error='Không tìm thấy tệp tải lên.')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('cvideo.html', error='Chưa chọn tệp để tải lên.')
        
        # Kiểm tra nếu tệp là video dựa trên định dạng
        if file and allowed_video(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Chuyển đổi video sang phong cách anime
            anime_video_path = convert_video_to_anime(filepath)
            anime_video_filename = os.path.basename(anime_video_path)
            
            return render_template('cvideo.html', original=filename, anime_video=anime_video_filename)
        else:
            return render_template('cvideo.html', error='Tệp tải lên không phải là định dạng video hợp lệ.')
    
    return render_template('cvideo.html')


# Trang giới thiệu
@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
