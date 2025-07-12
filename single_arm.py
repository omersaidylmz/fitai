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

def analyze_single_arm_cable(keypoints):
    """
    Single Arm Cable hareketini analiz eder
    Keypoints: 5-7-9 (sol kol) ve 6-8-10 (sağ kol)
    Yeni kriterler:
    - Başlangıç/Bitiş: Vücut açısı 15-25°, sol kol 100-115°
    - Perfect: Vücut açısı 90-105°
    """
    # Sol kol keypoints (omuz-dirsek-bilek)
    left_shoulder = keypoints[5][:2]   # Sol omuz
    left_elbow = keypoints[7][:2]      # Sol dirsek
    left_wrist = keypoints[9][:2]      # Sol bilek
    
    # Sağ kol keypoints (omuz-dirsek-bilek)
    right_shoulder = keypoints[6][:2]  # Sağ omuz
    right_elbow = keypoints[8][:2]     # Sağ dirsek
    right_wrist = keypoints[10][:2]    # Sağ bilek
    
    # Sol kalça keypoint
    left_hip = keypoints[12][:2]       # Sol kalça
    
    # Her iki kol için açıları hesapla
    left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    
    # Sağ dirsek - sağ omuz - sol kalça açısı (vücut açısı)
    body_angle = calculate_angle(right_elbow, right_shoulder, left_hip)
    
    # Ortalama açı (her iki kol için)
    avg_angle = (left_arm_angle + right_arm_angle) / 2
    
    # Hareket durumunu belirle
    movement_state = "unknown"
    feedback = ""
    is_correct = False
    
    # YENİ KRİTERLER:
    # Başlangıç/Bitiş pozisyonu: Vücut açısı 15-25° ve sol kol 100-115°
    if (15 <= body_angle <= 25) and (100 <= left_arm_angle <= 115):
        movement_state = "start_position"
        feedback = "Starting position - Ready to do Single Arm Cable"
        is_correct = True
    
    # Perfect curl pozisyonu: Vücut açısı 90-105°
    elif 90 <= body_angle <= 105:
        movement_state = "perfect_single_arm_cable"
        feedback = "Excellent Single Arm Cable position! Great!"
        is_correct = True
    
    # Geçiş durumları
    elif 25 < body_angle < 90:
        movement_state = "single_arm_cable_progress"
        feedback = "The Single Arm Cable movement continues"
        is_correct = True
    
    elif 35 < body_angle < 80:
        movement_state = "returning"
        feedback = "Başlangıç pozisyonuna dönüyor"
        is_correct = True
    
    # Hatalı pozisyonlar
    elif body_angle < 15:
        movement_state = "body_too_forward"
        feedback = "Body bent too far forward"
        is_correct = False
    
    elif body_angle > 115:
        movement_state = "body_too_back"
        feedback = "The body is bent too far back."
        is_correct = False
    
    # Sol kol açısı kontrolü (başlangıç pozisyonunda)
    elif (15 <= body_angle <= 25) and left_arm_angle < 100:
        movement_state = "arm_too_bent"
        feedback = "The left arm is bent too much, open it up."
        is_correct = False
    
    elif (15 <= body_angle <= 25) and left_arm_angle > 115:
        movement_state = "arm_too_straight"
        feedback = "Your left arm is too straight, bend it a little."
        is_correct = False
    
    else:
        movement_state = "adjustment_needed"
        feedback = "Position adjustment required"
        is_correct = False
    
    return feedback, is_correct, avg_angle, left_arm_angle, right_arm_angle, body_angle, movement_state

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
    ez_bar_count = 0  # EZ Bar curl sayacı
    last_state = "unknown"  # Son hareket durumu
    completed_curl = False  # Curl tamamlanma durumu
    
    # Çıktı ekranı için namedWindow oluştur ve boyutunu ayarla
    cv2.namedWindow("Single Arm Cable Analizi", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Single Arm Cable Analizi", 640, 480)
    
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
            
            if len(keypoints) >= 17:  # Gerekli keypoints varsa (sol kalça için 12 dahil)
                # EZ Bar curl analizi yap
                feedback, is_correct, avg_angle, left_angle, right_angle, body_angle, movement_state = analyze_single_arm_cable(keypoints)
                
                # YENİ TEKRAR SAYMA MANTĞI:
                # Perfect curl pozisyonuna ulaşınca completed_curl = True
                if movement_state == "perfect_single_arm_cable" and not completed_curl:
                    completed_curl = True
                # Vücut açısı 15-25° arasına dönünce tekrar sayısını artır
                elif movement_state == "start_position" and completed_curl:
                    ez_bar_count += 1
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
                
                # Vücut açısı çizgisi (8-6-12: sağ dirsek - sağ omuz - sol kalça)
                left_hip = tuple(map(int, keypoints[12][:2]))
                cv2.line(annotated_frame, right_elbow, right_shoulder, (255, 0, 255), 3)  # Mor
                cv2.line(annotated_frame, right_shoulder, left_hip, (255, 0, 255), 3)     # Mor
                
                # Analiz sonuçlarını ekrana yaz
                color = (0, 255, 0) if is_correct else (0, 0, 255)
                cv2.putText(annotated_frame, f"Feedback: {feedback}", (10, 30), font, 0.6, color, 2)
                cv2.putText(annotated_frame, f"Vucut Acisi: {body_angle:.1f}°", (10, 60), font, 0.6, (255, 0, 255), 2)
                cv2.putText(annotated_frame, f"Sol Kol: {left_angle:.1f}°", (10, 90), font, 0.6, (255, 0, 0), 2)
                cv2.putText(annotated_frame, f"Sag Kol: {right_angle:.1f}°", (10, 120), font, 0.6, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Ort. Kol Acisi: {avg_angle:.1f}°", (10, 150), font, 0.6, (255, 255, 0), 2)
                cv2.putText(annotated_frame, f"Single Arm Cable Sayisi: {ez_bar_count}", (10, 180), font, 0.7, (0, 255, 255), 2)
                cv2.putText(annotated_frame, f"Durum: {movement_state}", (10, 210), font, 0.6, (255, 255, 255), 2)
                
                # Yeni hedef açı aralıklarını göster
                cv2.putText(annotated_frame, "Baslangic: Vucut 15-25°, Sol kol 100-115°", (10, height-60), font, 0.5, (255, 255, 255), 1)
                cv2.putText(annotated_frame, "Perfect: Vucut 90-105°", (10, height-40), font, 0.5, (255, 255, 255), 1)
                cv2.putText(annotated_frame, "Mor cizgi: Vucut acisi (8-6-12)", (10, height-20), font, 0.5, (255, 0, 255), 1)
                
                last_state = movement_state
                
            else:
                annotated_frame = results[0].plot(labels=False, boxes=False)
                cv2.putText(annotated_frame, "Eksik keypoint tespiti", (10, 30), font, 0.7, (0, 0, 255), 2)
                cv2.putText(annotated_frame, f"Single Arm Cable Sayisi: {ez_bar_count}", (10, 60), font, 0.7, (0, 255, 255), 2)
        else:
            annotated_frame = frame.copy()
            cv2.putText(annotated_frame, "Kişi tespit edilemedi", (10, 30), font, 0.7, (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"Single Arm Cable Sayisi: {ez_bar_count}", (10, 60), font, 0.7, (0, 255, 255), 2)
        
        # Sonucu kaydet
        if output_path:
            out.write(annotated_frame)
        
        # Sonucu göster
        cv2.imshow("Single Arm Cable Analizi", annotated_frame)
        
        # 'q' tuşuna basılırsa çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Kaynakları serbest bırak
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()
    
    print(f"Toplam Single Arm Cable Sayisi: {ez_bar_count}")

def main():
    # Video yolu
    video_path = "single_arm_cable.mp4"  # Kendi video dosyanızın yolunu yazın
    output_path = "ez_bar_curl_output.mp4"  # Çıktı video dosyası
    
    # Video işlemesini başlat
    process_video(video_path, output_path)

if __name__ == "__main__":
    main()