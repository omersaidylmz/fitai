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

def analyze_cable_fly(keypoints, previous_avg_angle=None, direction="unknown"):
    """
    Standing Low to High Cable Fly hareketini analiz eder.
    Keypoints: Sol kol için 11 (sol kalça), 5 (sol omuz), 7 (sol dirsek)
               Sağ kol için 12 (sağ kalça), 6 (sağ omuz), 8 (sağ dirsek)
    Açı, omuz noktasında (5 veya 6) hesaplanır.
    previous_avg_angle: Bir önceki karedeki ortalama omuz açısı
    direction: Hareketin yönü ('up' (flexion/adduction) veya 'down' (extension/abduction))
    """
    # Sol kol keypoints (kalça-omuz-dirsek)
    left_hip = keypoints[11][:2]   # Sol kalça
    left_shoulder = keypoints[5][:2]   # Sol omuz
    left_elbow = keypoints[7][:2]      # Sol dirsek
    
    # Sağ kol keypoints (kalça-omuz-dirsek)
    right_hip = keypoints[12][:2]  # Sağ kalça
    right_shoulder = keypoints[6][:2]  # Sağ omuz
    right_elbow = keypoints[8][:2]     # Sağ dirsek
    
    # Her iki kol için kalça-omuz-dirsek açısını hesapla (omuz açısı)
    left_arm_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
    right_arm_angle = calculate_angle(right_hip, right_shoulder, right_elbow)
    
    # Ortalama açı (her iki kol için)
    avg_angle = (left_arm_angle + right_arm_angle) / 2
    
    # Hareket durumunu belirle
    movement_state = "unknown"
    feedback = ""
    is_correct = False
    
    # Hareket yönünü belirle (Cable Fly için: Açı küçülüyorsa yukarı/içeri (flexion/adduction), artıyorsa aşağı/dışarı (extension/abduction))
    current_direction = direction
    if previous_avg_angle is not None:
        if avg_angle < previous_avg_angle - 2: # Açı azalıyorsa (kollar yukarı/içeri geliyorsa) yukarı hareket
            current_direction = "up"
        elif avg_angle > previous_avg_angle + 2: # Açı artıyorsa (kollar aşağı/dışarı gidiyorsa) aşağı hareket
            current_direction = "down"
            
    # Başlangıç pozisyonu (Kollar aşağıda ve açık - geniş açı)
    # Omuz açısı: 90-120 derece (Hip-Shoulder-Elbow)
    if 45 <= avg_angle <= 50:
        movement_state = "start_position"
        feedback = "Kollar başlangıç pozisyonunda - Cable Fly yapmaya hazır"
        is_correct = True
    
    # Mükemmel Tepe Pozisyonu (Kollar yukarıda ve önde - dar açı)
    # Omuz açısı: 30-60 derece (Hip-Shoulder-Elbow)
    elif 25 <= avg_angle <= 30:
        movement_state = "perfect_cable_fly"
        feedback = "Mükemmel Cable Fly pozisyonu! Harika!"
        is_correct = True
    
    # Kollar yeterince yukarı gelmemiş veya çok açık (aşağı hareketten yukarıya)
    elif avg_angle > 35 and avg_angle < 40:
        movement_state = "need_more_adduction"
        if current_direction == "up": # Sadece yukarı hareket ederken bu geri bildirimi ver
            feedback = "Kolları daha fazla yukarı ve içeri getirin"
        else:
            feedback = "" # Aşağı inerken geri bildirim verme
        is_correct = False

    # Kollar çok fazla aşağıda veya çok açık (aşağı hareket)
    elif avg_angle > 120:
        movement_state = "too_wide_or_low"
        feedback = "Kollarınız çok açık veya çok aşağıda"
        is_correct = False
    
    # Kollar çok fazla yukarıda veya çok fazla içeri kapanmış
    elif avg_angle < 30:
        movement_state = "over_adducted_or_high"
        feedback = "Kollarınızı çok fazla yukarı/içeri kapattınız"
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
    fly_count = 0  # Cable Fly sayacı
    completed_fly_up = False  # Tepe noktasına ulaşıldı mı?
    
    previous_avg_angle = None # Bir önceki karedeki ortalama omuz açısı
    current_direction = "unknown" # Hareketin yönü ('down', 'up', 'unknown')

    # Çıktı ekranı için namedWindow oluştur ve boyutunu ayarla
    cv2.namedWindow("Cable Fly Analizi", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Cable Fly Analizi", 640, 480)
    
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
            
            # Gerekli keypointlerin (kalça, omuz, dirsek) mevcut olup olmadığını kontrol et
            # Sol kol için: 11 (kalça), 5 (omuz), 7 (dirsek)
            # Sağ kol için: 12 (kalça), 6 (omuz), 8 (dirsek)
            required_keypoints_indices = [11, 5, 7, 12, 6, 8]
            
            # **Hata düzeltmesi burada yapıldı**
            if all(idx < len(keypoints) for idx in required_keypoints_indices):
                # Cable Fly analizi yap
                feedback, is_correct, avg_angle, left_angle, right_angle, movement_state, new_direction = analyze_cable_fly(keypoints, previous_avg_angle, current_direction)
                
                # Hareket yönünü güncelle
                current_direction = new_direction

                # Tur sayma mantığı
                # Tepe noktasına ulaşıldığında (perfect_cable_fly) bayrağı ayarla
                if movement_state == "perfect_cable_fly" and not completed_fly_up:
                    completed_fly_up = True
                # Başlangıç pozisyonuna geri dönüldüğünde (start_position) ve tepe noktasına ulaşıldıysa sayacı artır
                elif movement_state == "start_position" and completed_fly_up:
                    fly_count += 1
                    completed_fly_up = False # Bayrağı sıfırla
                
                # Sadece sağ ve sol kolun ilgili keypointlerini çiz
                # Sol kol keypoints
                left_hip = tuple(map(int, keypoints[11][:2]))
                left_shoulder = tuple(map(int, keypoints[5][:2]))
                left_elbow = tuple(map(int, keypoints[7][:2]))
                
                # Sağ kol keypoints
                right_hip = tuple(map(int, keypoints[12][:2]))
                right_shoulder = tuple(map(int, keypoints[6][:2]))
                right_elbow = tuple(map(int, keypoints[8][:2]))

                # Keypointleri daire olarak çiz (sol kol mavi, sağ kol yeşil)
                cv2.circle(annotated_frame, left_hip, 5, (255, 0, 0), -1)
                cv2.circle(annotated_frame, left_shoulder, 5, (255, 0, 0), -1)
                cv2.circle(annotated_frame, left_elbow, 5, (255, 0, 0), -1)
                
                cv2.circle(annotated_frame, right_hip, 5, (0, 255, 0), -1)
                cv2.circle(annotated_frame, right_shoulder, 5, (0, 255, 0), -1)
                cv2.circle(annotated_frame, right_elbow, 5, (0, 255, 0), -1)

                # Kol çizgilerini çiz (sol kol mavi, sağ kol yeşil)
                cv2.line(annotated_frame, left_hip, left_shoulder, (255, 0, 0), 3)    # Mavi (kalça-omuz)
                cv2.line(annotated_frame, left_shoulder, left_elbow, (255, 0, 0), 3)  # Mavi (omuz-dirsek)
                
                cv2.line(annotated_frame, right_hip, right_shoulder, (0, 255, 0), 3)    # Yeşil (kalça-omuz)
                cv2.line(annotated_frame, right_shoulder, right_elbow, (0, 255, 0), 3)  # Yeşil (omuz-dirsek)
                
                # Analiz sonuçlarını ekrana yaz
                color = (0, 255, 0) if is_correct else (0, 0, 255)
                cv2.putText(annotated_frame, f"Feedback: {feedback}", (10, 30), font, 0.6, color, 2)
                cv2.putText(annotated_frame, f"Ort. Omuz Aci: {avg_angle:.1f}°", (10, 60), font, 0.6, (255, 255, 0), 2)
                cv2.putText(annotated_frame, f"Sol Kol Omuz Aci: {left_angle:.1f}°", (10, 90), font, 0.6, (255, 0, 0), 2)
                cv2.putText(annotated_frame, f"Sag Kol Omuz Aci: {right_angle:.1f}°", (10, 120), font, 0.6, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Cable Fly Sayisi: {fly_count}", (10, 150), font, 0.7, (0, 255, 255), 2)
                cv2.putText(annotated_frame, f"Durum: {movement_state}", (10, 180), font, 0.6, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"Yon: {current_direction}", (10, 210), font, 0.6, (255, 255, 255), 2) # Yön bilgisini ekle
                
                # Hedef açı aralığını göster
                cv2.putText(annotated_frame, "Hedef: 30-60° (Tepe), 90-120° (Baslangic)", (10, height-20), font, 0.5, (255, 255, 255), 1)
                
                # Bir sonraki kare için mevcut ortalama açıyı kaydet
                previous_avg_angle = avg_angle
                
            else:
                cv2.putText(annotated_frame, "Eksik keypoint tespiti (Kollar icin 5,6,7,8 ve kalca icin 11,12 gerekli)", (10, 30), font, 0.7, (0, 0, 255), 2)
                cv2.putText(annotated_frame, f"Cable Fly Sayisi: {fly_count}", (10, 60), font, 0.7, (0, 255, 255), 2)
                previous_avg_angle = None # Keypoint yoksa açıyı sıfırla
        else:
            annotated_frame = frame.copy()
            cv2.putText(annotated_frame, "Kişi tespit edilemedi", (10, 30), font, 0.7, (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"Cable Fly Sayisi: {fly_count}", (10, 60), font, 0.7, (0, 255, 255), 2)
            previous_avg_angle = None # Kişi yoksa açıyı sıfırla
        
        # Sonucu kaydet
        if output_path:
            out.write(annotated_frame)
        
        # Sonucu göster
        cv2.imshow("Cable Fly Analizi", annotated_frame)
        
        # 'q' tuşuna basılırsa çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Kaynakları serbest bırak
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()
    
    print(f"Toplam Cable Fly Sayisi: {fly_count}")

def main():
    # Video yolu
    video_path = "cable_fly.mp4"  # Kendi Standing Low to High Cable Fly video dosyanızın yolunu yazın
    output_path = "cable_fly_analysis.mp4"  # Çıktı video dosyası
    
    # Video işlemesini başlat
    process_video(video_path, output_path)

if __name__ == "__main__":
    main()