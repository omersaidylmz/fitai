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

def analyze_cable_lateral_raise(keypoints):
    """
    Cable Lateral Raise hareketini analiz eder.
    Keypoints: 6-8-10 (sağ kol) ve 5-7-9 (sol kol)
    Önemli olan 8 (sağ dirsek), 6 (sağ omuz), 12 (sağ kalça) arasındaki açı.
    Veya sol taraf için 7 (sol dirsek), 5 (sol omuz), 11 (sol kalça)
    """
    # Sağ kol keypoints (omuz-dirsek-bilek)
    right_shoulder = keypoints[6][:2]  # Sağ omuz
    right_elbow = keypoints[8][:2]     # Sağ dirsek
    right_hip = keypoints[12][:2]      # Sağ kalça
    
    # Sol kol keypoints (omuz-dirsek-bilek)
    left_shoulder = keypoints[5][:2]   # Sol omuz
    left_elbow = keypoints[7][:2]      # Sol dirsek
    left_hip = keypoints[11][:2]       # Sol kalça
    
    # Her iki kol için omuz-dirsek-kalça açısını hesapla
    # Açı 8 (dirsek), 6 (omuz), 12 (kalça) arasında olmalı
    right_arm_angle = calculate_angle(right_elbow, right_shoulder, right_hip)
    left_arm_angle = calculate_angle(left_elbow, left_shoulder, left_hip)
    
    # Ortalama açı (her iki kol için)
    avg_angle = (left_arm_angle + right_arm_angle) / 2
    
    # Hareket durumunu belirle
    movement_state = "unknown"
    feedback = ""
    is_correct = False
    
    # Başlangıç pozisyonu: kollar aşağıda, açı geniş (15-30 derece civarı)
    if 15 <= avg_angle <= 35:
        movement_state = "start_position"
        feedback = "Hareket başlangıç pozisyonu - Cable Lateral Raise yapmaya hazır"
        is_correct = True
    
    # Mükemmel lateral raise pozisyonu: kollar yanda, açı 80-100 derece
    elif 80 <= avg_angle <= 100:
        movement_state = "perfect_lateral_raise"
        feedback = "Mükemmel Cable Lateral Raise pozisyonu! Harika!"
        is_correct = True
    
    # Açı 30-80 arasında ise: daha fazla kaldırmalı
    elif 50 < avg_angle < 80:
        movement_state = "need_more_raise"
        feedback = "Biraz daha kolunuzu yukarı kaldırın"
        is_correct = False
    
    # Açı 100'den büyük ise: çok fazla kaldırılmış
    elif avg_angle > 100:
        movement_state = "over_raised"
        feedback = "Çok fazla kaldırdınız, biraz indirin"
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
    raise_count = 0  # Cable Lateral Raise sayacı
    completed_raise = False  # Raise tamamlanma durumu
    
    # Çıktı ekranı için namedWindow oluştur ve boyutunu ayarla
    cv2.namedWindow("Cable Lateral Raise Analizi", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Cable Lateral Raise Analizi", 640, 480)
    
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
            
            if len(keypoints) >= 13:  # Gerekli keypoints varsa (omuz, dirsek, kalça için 5,6,7,8,11,12)
                # Cable Lateral Raise analizi yap
                feedback, is_correct, avg_angle, left_angle, right_angle, movement_state = analyze_cable_lateral_raise(keypoints)
                
                # Tur sayma mantığı
                # Başlangıç pozisyonundan mükemmel pozisyona geçişte sayacı artır
                if movement_state == "perfect_lateral_raise" and not completed_raise:
                    completed_raise = True
                elif movement_state == "start_position" and completed_raise:
                    raise_count += 1
                    completed_raise = False
                
                # Keypoints'leri çiz
                annotated_frame = results[0].plot(labels=False, boxes=False)
                
                # Kol açılarını görselleştir
                # Sol kol çizgileri (omuz-dirsek ve omuz-kalça)
                left_shoulder = tuple(map(int, keypoints[5][:2]))
                left_elbow = tuple(map(int, keypoints[7][:2]))
                left_hip = tuple(map(int, keypoints[11][:2]))
                
                cv2.line(annotated_frame, left_shoulder, left_elbow, (255, 0, 0), 3)  # Mavi (omuz-dirsek)
                cv2.line(annotated_frame, left_shoulder, left_hip, (255, 0, 0), 3)    # Mavi (omuz-kalça)
                
                # Sağ kol çizgileri (omuz-dirsek ve omuz-kalça)
                right_shoulder = tuple(map(int, keypoints[6][:2]))
                right_elbow = tuple(map(int, keypoints[8][:2]))
                right_hip = tuple(map(int, keypoints[12][:2]))
                
                cv2.line(annotated_frame, right_shoulder, right_elbow, (0, 255, 0), 3)  # Yeşil (omuz-dirsek)
                cv2.line(annotated_frame, right_shoulder, right_hip, (0, 255, 0), 3)    # Yeşil (omuz-kalça)
                
                # Analiz sonuçlarını ekrana yaz
                color = (0, 255, 0) if is_correct else (0, 0, 255)
                cv2.putText(annotated_frame, f"Feedback: {feedback}", (10, 30), font, 0.6, color, 2)
                cv2.putText(annotated_frame, f"Ort. Aci: {avg_angle:.1f}°", (10, 60), font, 0.6, (255, 255, 0), 2)
                cv2.putText(annotated_frame, f"Sol Kol Acisi: {left_angle:.1f}°", (10, 90), font, 0.6, (255, 0, 0), 2)
                cv2.putText(annotated_frame, f"Sag Kol Acisi: {right_angle:.1f}°", (10, 120), font, 0.6, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Cable Lateral Raise Sayisi: {raise_count}", (10, 150), font, 0.7, (0, 255, 255), 2)
                cv2.putText(annotated_frame, f"Durum: {movement_state}", (10, 180), font, 0.6, (255, 255, 255), 2)
                
                # Hedef açı aralığını göster
                cv2.putText(annotated_frame, "Hedef: 80-100° (Kaldirma), 15-30° (Baslangic)", (10, height-20), font, 0.5, (255, 255, 255), 1)
                
            else:
                annotated_frame = results[0].plot(labels=False, boxes=False)
                cv2.putText(annotated_frame, "Eksik keypoint tespiti", (10, 30), font, 0.7, (0, 0, 255), 2)
                cv2.putText(annotated_frame, f"Cable Lateral Raise Sayisi: {raise_count}", (10, 60), font, 0.7, (0, 255, 255), 2)
        else:
            annotated_frame = frame.copy()
            cv2.putText(annotated_frame, "Kişi tespit edilemedi", (10, 30), font, 0.7, (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"Cable Lateral Raise Sayisi: {raise_count}", (10, 60), font, 0.7, (0, 255, 255), 2)
        
        # Sonucu kaydet
        if output_path:
            out.write(annotated_frame)
        
        # Sonucu göster
        cv2.imshow("Cable Lateral Raise Analizi", annotated_frame)
        
        # 'q' tuşuna basılırsa çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Kaynakları serbest bırak
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()
    
    print(f"Toplam Cable Lateral Raise Sayisi: {raise_count}")

def main():
    # Video yolu
    video_path = "cable_lateral.mp4"  # Kendi video dosyanızın yolunu yazın
    output_path = "cable_lateral_raise_output.mp4"  # Çıktı video dosyası
    
    # Video işlemesini başlat
    process_video(video_path, output_path)

if __name__ == "__main__":
    main()