import cv2
from ultralytics import YOLO
import numpy as np

# YOLOv8 Pose modelini yükle
model = YOLO('yolov8n-pose.pt')

def calculate_angle(a, b, c):
    """
    Üç nokta arasındaki açıyı hesaplar
    a, b, c: numpy dizileri [x, y] formatında
    b noktası, açının tepe noktasıdır
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    # Vektörleri oluştur
    ba = a - b
    bc = c - b
    
    # Açıyı kosinüs teoremi ile hesapla
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    # Radyandan dereceye çevir
    angle_degrees = np.degrees(angle)
    
    return angle_degrees

def analyze_seated_dumbell(keypoints):
    """
    Seated Dumbell hareketini analiz eder
    Keypoints: 5-7-9 (sol kol) ve 6-8-10 (sağ kol)
    """
    # Sol kol keypoints (omuz-dirsek-bilek)
    left_shoulder = keypoints[5][:2]   # Sol omuz
    left_elbow = keypoints[7][:2]      # Sol dirsek
    left_wrist = keypoints[9][:2]      # Sol bilek
    
    # Sağ kol keypoints (omuz-dirsek-bilek)
    right_shoulder = keypoints[6][:2]  # Sağ omuz
    right_elbow = keypoints[8][:2]     # Sağ dirsek
    right_wrist = keypoints[10][:2]    # Sağ bilek
    
    # Her iki kol için açıları hesapla
    left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    
    # Ortalama açı (her iki kol için)
    avg_angle = (left_arm_angle + right_arm_angle) / 2
    
    # Hareket durumunu belirle
    movement_state = "unknown"
    feedback = ""
    is_correct = False
    
    # 160 derece civarında - hareketin başlangıcı/sonu
    if 55 <= avg_angle <= 70:
        movement_state = "start_position"
        feedback = "Starting position - Ready to perform seated dumbbell exercises"
        is_correct = True
    
    # 30-45 derece - mükemmel curl pozisyonu
    elif 145 <= avg_angle <= 165:
        movement_state = "perfect_seated_dumbell"
        feedback = "Perfect Seated Dumbbell position! Great!"
        is_correct = True
    
    # 45 dereceden büyük - daha fazla bükmeli
    elif 115 < avg_angle < 135:
        movement_state = "need_more_seated_dumbell"
        feedback = "Push your arm a little further"
        is_correct = False
    
    # 30 dereceden küçük - çok fazla bükülmüş
    elif avg_angle < 170:
        movement_state = "over_seated"
        feedback = "You bent it too much."
        is_correct = False
    
    return feedback, is_correct, avg_angle, left_arm_angle, right_arm_angle, movement_state

def process_video(video_path, output_path=None):
    # Video dosyasını aç
    cap = cv2.VideoCapture(video_path)
    
    # Videonun özelliklerini al
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Çıkış videosu için codec ve writer oluştur
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Yazı fontunu ayarla
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    frame_count = 0
    curl_count = 0  # Cable curl sayacı
    last_state = "unknown"  # Son hareket durumu
    completed_curl = False  # Curl tamamlanma durumu
    
    # Çıktı ekranı için namedWindow oluştur ve boyutunu ayarla
    cv2.namedWindow("Seated Dumbell Analizi", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Seated Dumbell Analizi", 640, 480)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        
        # İşleme süresini azaltmak için her 2 karede bir işleme yapabilirsiniz
        if frame_count % 2 != 0:
            continue
        
        # YOLOv8 ile pose tahmini yap
        results = model(frame, classes=None)
        
        if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
            # Keypoints'leri al
            keypoints = results[0].keypoints.data[0].cpu().numpy()
            
            if len(keypoints) >= 11:  # Gerekli keypoints varsa
                # Cable curl analizi yap
                feedback, is_correct, avg_angle, left_angle, right_angle, movement_state = analyze_seated_dumbell(keypoints)
                
                # Tur sayma mantığı
                if movement_state == "perfect_seated_dumbell" and not completed_curl:
                    completed_curl = True
                elif movement_state == "start_position" and completed_curl:
                    curl_count += 1
                    completed_curl = False
                
                # Keypoints'leri çiz
                annotated_frame = results[0].plot(labels=False, boxes=False)
                
                # Kol açılarını görselleştir
                # Sol kol çizgileri
                left_shoulder = tuple(map(int, keypoints[5][:2]))
                left_elbow = tuple(map(int, keypoints[7][:2]))
                left_wrist = tuple(map(int, keypoints[9][:2]))
                
                cv2.line(annotated_frame, left_shoulder, left_elbow, (255, 0, 0), 3)  # Mavi
                cv2.line(annotated_frame, left_elbow, left_wrist, (255, 0, 0), 3)
                
                # Sağ kol çizgileri
                right_shoulder = tuple(map(int, keypoints[6][:2]))
                right_elbow = tuple(map(int, keypoints[8][:2]))
                right_wrist = tuple(map(int, keypoints[10][:2]))
                
                cv2.line(annotated_frame, right_shoulder, right_elbow, (0, 255, 0), 3)  # Yeşil
                cv2.line(annotated_frame, right_elbow, right_wrist, (0, 255, 0), 3)
                
                # Analiz sonuçlarını ekrana yaz
                color = (0, 255, 0) if is_correct else (0, 0, 255)
                cv2.putText(annotated_frame, f"Feedback: {feedback}", (10, 30), font, 0.6, color, 2)
                cv2.putText(annotated_frame, f"Ort. Aci: {avg_angle:.1f}°", (10, 60), font, 0.6, (255, 255, 0), 2)
                cv2.putText(annotated_frame, f"Sol Kol: {left_angle:.1f}°", (10, 90), font, 0.6, (255, 0, 0), 2)
                cv2.putText(annotated_frame, f"Sag Kol: {right_angle:.1f}°", (10, 120), font, 0.6, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Seated Dumbell  Sayisi: {curl_count}", (10, 150), font, 0.7, (0, 255, 255), 2)
                cv2.putText(annotated_frame, f"Durum: {movement_state}", (10, 180), font, 0.6, (255, 255, 255), 2)
                
                # Hedef açı aralığını göster
                cv2.putText(annotated_frame, "Hedef: 145-165° (Curl), 55+° (Baslangic)", (10, height-20), font, 0.5, (255, 255, 255), 1)
                
                last_state = movement_state
                
            else:
                annotated_frame = results[0].plot(labels=False, boxes=False)
                cv2.putText(annotated_frame, "Eksik keypoint tespiti", (10, 30), font, 0.7, (0, 0, 255), 2)
                cv2.putText(annotated_frame, f"Seated Dumbell  Sayisi: {curl_count}", (10, 60), font, 0.7, (0, 255, 255), 2)
        else:
            annotated_frame = frame.copy()
            cv2.putText(annotated_frame, "Kişi tespit edilemedi", (10, 30), font, 0.7, (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"Seated Dumbell Sayisi: {curl_count}", (10, 60), font, 0.7, (0, 255, 255), 2)
        
        # Sonucu kaydet
        if output_path:
            out.write(annotated_frame)
        
        # Sonucu göster
        cv2.imshow("Pulley Cable Analizi", annotated_frame)
        
        # 'q' tuşuna basılırsa çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Kaynakları serbest bırak
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()
    
    print(f"Toplam Pulley Cable Sayisi: {curl_count}")

def main():
    # Video yolu
    video_path = "seated_dumbell.mp4"  # Kendi video dosyanızın yolunu yazın
    output_path = "cable_curl_output.mp4"  # Çıktı video dosyası
    
    # Video işlemesini başlat
    process_video(video_path, output_path)

if __name__ == "__main__":
    main()