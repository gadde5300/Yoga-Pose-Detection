import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle

def predict(frame,output_directory):
    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        image = cv2.imread(frame)
        height, width = image.shape[:2] #getting the shape of the image.
        # Recolor image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks

        landmarks = results.pose_landmarks.landmark
        # Get coordinates
        try:
            hip_left= [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee_left = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            foot_left = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
            angle_left = calculate_angle(hip_left, knee_left, foot_left)
        except:
            pass

        try:
            hip_right= [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            knee_right = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            foot_right = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
            angle_right = calculate_angle(hip_right, knee_right, foot_right)

        except:
            pass


        try:
            shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y] 
            hip_left= [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            foot_left = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
            angle_downward = calculate_angle(shoulder_left, hip_left, foot_left)


        except:
            pass

    #     print(angle_downward,angle_right,angle_left)
        if int(angle_downward) < 90:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                             )   


            ans = str(int(angle_downward))+" "+ "degrees"
            cv2.putText(image, ans, 
                       tuple(np.multiply(knee_right, [width,height]).astype(int)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7,(80,80,80), 2, cv2.LINE_AA
                            )
            cv2.rectangle(image, (0,0), (300, 50), (0, 0, 0), -1)

                # Display Class
            cv2.putText(image, 'Downward-Facing Dog Pose', (8,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(192, 192, 192),2,cv2.LINE_AA)
            output_file = output_directory + "image.jpg"
            cv2.imwrite(output_file,image)
            return output_file


    #     print(angle_left)
        elif int(angle_right)<90 or int(angle_left)<90:

            if int(angle_right)<90:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                         )               
                ans = str(int(angle_right))+" "+"degrees"
                cv2.putText(image, ans, 
                           tuple(np.multiply(knee_right, [width,height]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7,(80,80,80), 2, cv2.LINE_AA
                            )

                cv2.rectangle(image, (0,0), (240, 50), (0, 0, 0), -1)

                # Display Class
                cv2.putText(image, 'Tree Pose'
                        , (8,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (192, 192, 192), 2,cv2.LINE_AA)
                output_file = output_directory + "image.jpg"
                cv2.imwrite(output_file,image)
                return output_file

            elif int(angle_left)<90:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                 )               
                ans = str(int(angle_left))+" "+"degrees"
                cv2.putText(image, ans, 
                           tuple(np.multiply(knee_left, [width,height]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80,80,80), 2, cv2.LINE_AA
                                )
                cv2.rectangle(image, (0,0), (240, 50), (0, 0, 0), -1)

                # Display Class
                cv2.putText(image, 'Tree Pose'
                        , (8,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (192, 192, 192), 2,cv2.LINE_AA)
                output_file = output_directory + "image.jpg"
                cv2.imwrite(output_file,image)
                return output_file

        elif (int(angle_right)>170 or int(angle_left)>170) and int(angle_downward)>100 :

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                     )               
            ans= str(int(angle_downward))+" "+"degrees"
            cv2.putText(image, ans, 
                       tuple(np.multiply(knee_right, [width,height]).astype(int)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80,80,80), 2, cv2.LINE_AA
                            )
            cv2.rectangle(image, (0,0), (240, 50), (0, 0, 0), -1)

                # Display Class
            cv2.putText(image, 'Mountain Pose'
                        , (8,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (192, 192, 192), 2,cv2.LINE_AA)
            output_file = output_directory + "image.jpg"
            cv2.imwrite(output_file,image)
            return output_file
