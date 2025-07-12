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

def analyze_incline_press(keypoints, previous_avg_angle=None, direction="unknown"):
    """
    Incline Press hareketini analiz eder.
    Keypoints: Sol kol için 5 (omuz), 7 (dirsek), 9 (bilek)
               Sağ kol için 6 (omuz), 8 (dirsek), 10 (bilek)
    Açı, dirsek noktasında (7 veya 8) hesaplanır.
    previous_avg_angle: Bir önceki karedeki ortalama dirsek açısı
    direction: Hareketin yönü ('down' (flexion) veya 'up' (extension))
    """
    # Sol kol keypoints (omuz-dirsek-bilek)
    left_shoulder = keypoints[5][:2]  # Sol omuz
    left_elbow = keypoints[7][:2]    # Sol dirsek
    left_wrist = keypoints[9][:2]    # Sol bilek
    
    # Sağ kol keypoints (omuz-dirsek-bilek)
    right_shoulder = keypoints[6][:2] # Sağ omuz
    right_elbow = keypoints[8][:2]   # Sağ dirsek
    right_wrist = keypoints[10][:2]   # Sağ bilek
    
    # Her iki kol için omuz-dirsek-bilek açısını hesapla
    left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    
    # Ortalama açı (her iki kol için)
    avg_angle = (left_arm_angle + right_arm_angle) / 2
    
    # Hareket durumunu belirle
    movement_state = "unknown"
    feedback = ""
    is_correct = False
    
    # Hareket yönünü belirle (Incline Press için: Açı artıyorsa yukarı (extension), azalıyorsa aşağı (flexion))
    current_direction = direction
    if previous_avg_angle is not None:
        if avg_angle > previous_avg_angle + 2: # Açı artıyorsa (kollar uzuyorsa) yukarı hareket (extension)
            current_direction = "up"
        elif avg_angle < previous_avg_angle - 2: # Açı azalıyorsa (kollar bükülüyorsa) aşağı hareket (flexion)
            current_direction = "down"
            
    # Başlangıç pozisyonu (Bar yukarıda - tepe noktası)
    # Dirsek açısı geniş: 135-145 derece
    if 135 <= avg_angle <= 145:
        movement_state = "start_position"
        feedback = "Arms extended - Ready to perform an incline press"
        is_correct = True
    
    # Mükemmel Incline Press pozisyonu (Bar göğüse yakın - dip noktası)
    # Dirsek açısı dar: 35-45 derece
    elif 35 <= avg_angle <= 45:
        movement_state = "perfect_incline_press"
        feedback = "Excellent Incline Press depth! Great!"
        is_correct = True
    
    # Dirsek açısı çok geniş (daha derine inmeli - eğer aşağı gidiyorsa)
    elif avg_angle > 70 and avg_angle < 115: # Açı 45-135 arası
        movement_state = "intermediate" # Ara durum
        if current_direction == "down": # Sadece aşağı hareket ederken bu geri bildirimi ver
            feedback = "Go a little deeper"
        elif current_direction == "up": # Yukarı hareket ederken kolları uzatması gerektiğini söyle
            feedback = "Extend your arms further"
        else:
            feedback = "" # Dururken veya yön belirsizken geri bildirim verme
        is_correct = False
    
    # Dirsek açısı çok geniş (aşırı gerilmiş veya pozisyon bozuk)
    elif avg_angle > 150:
        movement_state = "over_extended"
        feedback = "You stretched your arms too much."
        is_correct = False

    # Dirsek açısı çok dar (çok fazla indiniz)
    elif avg_angle < 35:
        movement_state = "too_deep"
        feedback = "You've bent too much, watch your elbows."
        is_correct = False
    
    return feedback, is_correct, avg_angle, left_arm_angle, right_arm_angle, movement_state, current_direction

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
    incline_press_count = 0  # Incline Press sayacı
    completed_press_down = False  # Dip noktasına ulaşıldı mı?
    
    previous_avg_angle = None # Bir önceki karedeki ortalama dirsek açısı
    current_direction = "unknown" # Hareketin yönü ('down', 'up', 'unknown')

    # Çıktı ekranı için namedWindow oluştur ve boyutunu ayarla
    cv2.namedWindow("Incline Press Analizi", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Incline Press Analizi", 640, 480)
    
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
        
        # Yeni boş çerçeve oluştur, üzerine sadece ilgili keypointleri çizeceğiz
        annotated_frame = frame.copy() 

        if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
            # Keypoints'leri al
            keypoints = results[0].keypoints.data[0].cpu().numpy()
            
            # Gerekli keypointlerin (omuz, dirsek, bilek) mevcut olup olmadığını kontrol et
            # Sol kol için: 5 (omuz), 7 (dirsek), 9 (bilek)
            # Sağ kol için: 6 (omuz), 8 (dirsek), 10 (bilek)
            required_keypoints_indices = [5, 6, 7, 8, 9, 10]
            
            # Tüm gerekli keypointlerin varlığını kontrol et
            if all(idx < len(keypoints) for idx in required_keypoints_indices):
                # Incline Press analizi yap
                feedback, is_correct, avg_angle, left_angle, right_angle, movement_state, new_direction = analyze_incline_press(keypoints, previous_avg_angle, current_direction)
                
                # Hareket yönünü güncelle
                current_direction = new_direction

                # Tur sayma mantığı
                # Dip noktasına ulaşıldığında (perfect_incline_press) bayrağı ayarla
                if movement_state == "perfect_incline_press":
                    completed_press_down = True
                # Tepe noktasına geri dönüldüğünde (start_position) ve dip noktasına ulaşıldıysa sayacı artır
                elif movement_state == "start_position" and completed_press_down:
                    incline_press_count += 1
                    completed_press_down = False # Bayrağı sıfırla
                
                # Sadece sağ ve sol kol keypointlerini çiz
                # Sol kol keypoints
                left_shoulder = tuple(map(int, keypoints[5][:2]))
                left_elbow = tuple(map(int, keypoints[7][:2]))
                left_wrist = tuple(map(int, keypoints[9][:2]))
                
                # Sağ kol keypoints
                right_shoulder = tuple(map(int, keypoints[6][:2]))
                right_elbow = tuple(map(int, keypoints[8][:2]))
                right_wrist = tuple(map(int, keypoints[10][:2]))

                # Keypointleri daire olarak çiz (sol kol mavi, sağ kol yeşil)
                cv2.circle(annotated_frame, left_shoulder, 5, (255, 0, 0), -1)
                cv2.circle(annotated_frame, left_elbow, 5, (255, 0, 0), -1)
                cv2.circle(annotated_frame, left_wrist, 5, (255, 0, 0), -1)
                
                cv2.circle(annotated_frame, right_shoulder, 5, (0, 255, 0), -1)
                cv2.circle(annotated_frame, right_elbow, 5, (0, 255, 0), -1)
                cv2.circle(annotated_frame, right_wrist, 5, (0, 255, 0), -1)

                # Kol çizgilerini çiz (sol kol mavi, sağ kol yeşil)
                cv2.line(annotated_frame, left_shoulder, left_elbow, (255, 0, 0), 3)    # Mavi (omuz-dirsek)
                cv2.line(annotated_frame, left_elbow, left_wrist, (255, 0, 0), 3)  # Mavi (dirsek-bilek)
                
                cv2.line(annotated_frame, right_shoulder, right_elbow, (0, 255, 0), 3)    # Yeşil (omuz-dirsek)
                cv2.line(annotated_frame, right_elbow, right_wrist, (0, 255, 0), 3)  # Yeşil (dirsek-bilek)
                
                # Analiz sonuçlarını ekrana yaz
                color = (0, 255, 0) if is_correct else (0, 0, 255)
                cv2.putText(annotated_frame, f"Feedback: {feedback}", (10, 30), font, 0.6, color, 2)
                cv2.putText(annotated_frame, f"Ort. Dirsek Aci: {avg_angle:.1f}°", (10, 60), font, 0.6, (255, 255, 0), 2)
                cv2.putText(annotated_frame, f"Sol Dirsek Aci: {left_angle:.1f}°", (10, 90), font, 0.6, (255, 0, 0), 2)
                cv2.putText(annotated_frame, f"Sag Dirsek Aci: {right_angle:.1f}°", (10, 120), font, 0.6, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Incline Press Sayisi: {incline_press_count}", (10, 150), font, 0.7, (0, 255, 255), 2)
                cv2.putText(annotated_frame, f"Durum: {movement_state}", (10, 180), font, 0.6, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"Yon: {current_direction}", (10, 210), font, 0.6, (255, 255, 255), 2) # Yön bilgisini ekle
                
                # Hedef açı aralığını göster
                cv2.putText(annotated_frame, "Hedef: 35-45° (Dip), 135-145° (Tepe)", (10, height-20), font, 0.5, (255, 255, 255), 1)
                
                # Bir sonraki kare için mevcut ortalama açıyı kaydet
                previous_avg_angle = avg_angle
                
            else:
                cv2.putText(annotated_frame, "Eksik keypoint tespiti (Kollar icin 5,6,7,8,9,10 gerekli)", (10, 30), font, 0.7, (0, 0, 255), 2)
                cv2.putText(annotated_frame, f"Incline Press Sayisi: {incline_press_count}", (10, 60), font, 0.7, (0, 255, 255), 2)
                previous_avg_angle = None # Keypoint yoksa açıyı sıfırla
        else:
            cv2.putText(annotated_frame, "Kişi tespit edilemedi", (10, 30), font, 0.7, (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"Incline Press Sayisi: {incline_press_count}", (10, 60), font, 0.7, (0, 255, 255), 2)
            previous_avg_angle = None # Kişi yoksa açıyı sıfırla
        
        # Sonucu kaydet
        if output_path:
            out.write(annotated_frame)
        
        # Sonucu göster
        cv2.imshow("Incline Press Analizi", annotated_frame)
        
        # 'q' tuşuna basılırsa çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Kaynakları serbest bırak
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()
    
    print(f"Toplam Incline Press Sayisi: {incline_press_count}")

def main():
    # Video yolu
    video_path = "incline_press.mp4"  # Kendi incline press video dosyanızın yolunu yazın
    output_path = "incline_press_analysis.mp4"  # Çıktı video dosyası
    
    # Video işlemesini başlat
    process_video(video_path, output_path)

if __name__ == "__main__":
    main()