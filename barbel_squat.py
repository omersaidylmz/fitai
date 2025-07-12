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

def analyze_squat(keypoints, previous_leg_angle=None, direction="unknown"):
    """
    Squat hareketini analiz eder.
    Keypoints: Sol bacak için 11 (kalça), 13 (diz), 15 (ayak bileği)
    Açı, sol diz noktasında (13) hesaplanır.
    previous_leg_angle: Bir önceki karedeki diz açısı
    direction: Hareketin yönü ('down' veya 'up')
    """
    # Sol bacak keypoints (kalça-diz-ayak bileği)
    left_hip = keypoints[11][:2]   # Sol kalça
    left_knee = keypoints[13][:2]  # Sol diz
    left_ankle = keypoints[15][:2] # Sol ayak bileği
    
    # Sol bacak için kalça-diz-ayak bileği açısını hesapla
    # Açı diz noktasında hesaplanır
    leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
    
    # Hareket durumunu belirle
    movement_state = "unknown"
    feedback = ""
    is_correct = False
    
    # Hareket yönünü belirle
    current_direction = direction
    if previous_leg_angle is not None:
        if leg_angle < previous_leg_angle - 2: # Açı azalıyorsa (diz bükülüyorsa) aşağı hareket
            current_direction = "down"
        elif leg_angle > previous_leg_angle + 2: # Açı artıyorsa (diz açılıyorsa) yukarı hareket
            current_direction = "up"
            
    # Başlangıç pozisyonu: Dizler neredeyse düz (160-175 derece)
    if 160 <= leg_angle <= 175:
        movement_state = "start_position"
        feedback = "Ayakta duruş pozisyonu - Squat yapmaya hazır"
        is_correct = True
    
    # Mükemmel squat pozisyonu: Dizler yaklaşık 90 derece (75-95 derece)
    elif 75 <= leg_angle <= 95:
        movement_state = "perfect_squat"
        feedback = "Mükemmel Squat derinliği! Harika!"
        is_correct = True
    
    # Diz açısı çok geniş (daha derine inmeli)
    elif leg_angle > 95 and leg_angle < 160:
        movement_state = "need_deeper"
        if current_direction == "down": # Sadece aşağı hareket ederken bu geri bildirimi ver
            feedback = "Biraz daha derine inin"
        else: # Yukarı kalkarken veya dururken geri bildirim verme
            feedback = ""
        is_correct = False
    
    # Diz açısı çok dar (çok fazla derine inilmiş veya pozisyon bozuk)
    elif leg_angle < 75:
        movement_state = "too_deep_or_incorrect"
        feedback = "Çok fazla derine indiniz veya dizlerinizde sorun var"
        is_correct = False
    
    # Fonksiyonun orijinal çıktısıyla uyumlu olması için diğer değerler None
    return feedback, is_correct, leg_angle, leg_angle, None, movement_state, current_direction # avg_angle ve left_angle aynı, right_angle None

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
    squat_count = 0  # Squat sayacı
    completed_squat = False  # Squat tamamlanma durumu
    
    previous_leg_angle = None # Bir önceki karedeki diz açısı
    current_direction = "unknown" # Hareketin yönü ('down', 'up', 'unknown')

    # Çıktı ekranı için namedWindow oluştur ve boyutunu ayarla
    cv2.namedWindow("Squat Analizi", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Squat Analizi", 640, 480)
    
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
            
            # Gerekli keypoints (sol kalça, sol diz, sol ayak bileği) mevcut mu kontrol et
            if len(keypoints) >= 16:  # Keypoint'ler 0'dan 15'e kadar olduğu için en az 16 keypoint olmalı
                # Squat analizi yap
                feedback, is_correct, avg_angle, left_angle, right_angle, movement_state, new_direction = analyze_squat(keypoints, previous_leg_angle, current_direction)
                
                # Hareket yönünü güncelle
                current_direction = new_direction

                # Tur sayma mantığı
                # Başlangıç pozisyonundan mükemmel pozisyona geçişte sayacı artır
                if movement_state == "perfect_squat" and not completed_squat:
                    completed_squat = True
                elif movement_state == "start_position" and completed_squat:
                    squat_count += 1
                    completed_squat = False
                
                # Keypoints'leri çiz
                annotated_frame = results[0].plot(labels=False, boxes=False)
                
                # Sadece sol bacak açılarını görselleştir
                # Sol bacak çizgileri (kalça-diz-ayak bileği)
                left_hip = tuple(map(int, keypoints[11][:2]))
                left_knee = tuple(map(int, keypoints[13][:2]))
                left_ankle = tuple(map(int, keypoints[15][:2]))
                
                cv2.line(annotated_frame, left_hip, left_knee, (255, 0, 0), 3)    # Mavi (kalça-diz)
                cv2.line(annotated_frame, left_knee, left_ankle, (255, 0, 0), 3)  # Mavi (diz-ayak bileği)
                
                # Analiz sonuçlarını ekrana yaz
                color = (0, 255, 0) if is_correct else (0, 0, 255)
                cv2.putText(annotated_frame, f"Feedback: {feedback}", (10, 30), font, 0.6, color, 2)
                cv2.putText(annotated_frame, f"Sol Diz Aci: {left_angle:.1f}°", (10, 60), font, 0.6, (255, 0, 0), 2)
                cv2.putText(annotated_frame, f"Squat Sayisi: {squat_count}", (10, 90), font, 0.7, (0, 255, 255), 2)
                cv2.putText(annotated_frame, f"Durum: {movement_state}", (10, 120), font, 0.6, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"Yon: {current_direction}", (10, 150), font, 0.6, (255, 255, 255), 2) # Yön bilgisini ekle
                
                # Hedef açı aralığını göster
                cv2.putText(annotated_frame, "Hedef: 75-95° (Squat), 160-175° (Baslangic)", (10, height-20), font, 0.5, (255, 255, 255), 1)
                
                # Bir sonraki kare için mevcut açıyı kaydet
                previous_leg_angle = left_angle
                
            else:
                annotated_frame = results[0].plot(labels=False, boxes=False)
                cv2.putText(annotated_frame, "Eksik keypoint tespiti (Sol bacak icin 11,13,15 gerekli)", (10, 30), font, 0.7, (0, 0, 255), 2)
                cv2.putText(annotated_frame, f"Squat Sayisi: {squat_count}", (10, 60), font, 0.7, (0, 255, 255), 2)
                previous_leg_angle = None # Keypoint yoksa açıyı sıfırla
        else:
            annotated_frame = frame.copy()
            cv2.putText(annotated_frame, "Kişi tespit edilemedi", (10, 30), font, 0.7, (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"Squat Sayisi: {squat_count}", (10, 60), font, 0.7, (0, 255, 255), 2)
            previous_leg_angle = None # Kişi yoksa açıyı sıfırla
        
        # Sonucu kaydet
        if output_path:
            out.write(annotated_frame)
        
        # Sonucu göster
        cv2.imshow("Squat Analizi", annotated_frame)
        
        # 'q' tuşuna basılırsa çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Kaynakları serbest bırak
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()
    
    print(f"Toplam Squat Sayisi: {squat_count}")

def main():
    # Video yolu
    video_path = "barbel_squat.mp4"  # Kendi video dosyanızın yolunu yazın
    output_path = "squat_output_left_leg_directional.mp4"  # Çıktı video dosyası
    
    # Video işlemesini başlat
    process_video(video_path, output_path)

if __name__ == "__main__":
    main()