import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np 
from moviepy.editor import *


#Calibrate the camera
images=[]
base_path='/Users/johnmcconnell/Desktop/SDC Nano Degree/CarND-Advanced-Lane-Lines-master/camera_cal/calibration'
suffix='.jpg'
for i in range(1,21):
    i=str(i)
    img=base_path+i+suffix
    images.append(img)

#get object and image points
objpoints=[]
imgpoints=[]
objp=np.zeros((6*9,3),np.float32)
objp[:,:2]=np.mgrid[0:9,0:6].T.reshape(-1,2)
for fname in images:
    img=cv2.imread(fname)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    if ret==True:
        imgpoints.append(corners)
        objpoints.append(objp)

#run calibrate camera function from cv2
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

def warp(img):
    #a function to warp images based on target points
    #get image size
    img_size=(img.shape[1],img.shape[0])
    #four points, src and dst
    src=np.float32([[235,720],
                    [1132,720],
                    [603,450],
                    [720,450]])
    dst=np.float32([[300,0],
                    [900,0],
                    [300,720],
                    [900,720]]) 
    #compute matrix
    M = cv2.getPerspectiveTransform(src, dst)
    #compute image
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    #return image
    return warped

def un_warp(img):
    #a function to unwarp images based on target points
    #get image size
    img_size=(img.shape[1],img.shape[0])
    #four points, src and dst
    src=np.float32([[235,720],
                    [1132,720],
                    [603,450],
                    [720,450]])
    dst=np.float32([[300,0],
                    [900,0],
                    [300,720],
                    [900,720]])
    #inverse matrix
    M = cv2.getPerspectiveTransform(dst, src)
    #un warp image
    un_warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    #return image
    return un_warped

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = int(255)
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def find_lines(image_1):
    #a function to detect the lines on the image, only the first image

    #Import the mtx and dist varibles
    global mtx, dist
    #read in image, only when in picture mode
    #img=cv2.imread(image_1)

    #Undistort image
    img = cv2.undistort(image_1, mtx, dist, None, mtx)

    #create a layover image for later
    layover_image=img

    #warp image
    img=warp(img)

    #Convert to HLS
    hls=cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    
    #go to just s channel
    s_channel = hls[:,:,2]
    
    #threshold s channel 
    s_thresh_min = 170
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    #Create gray image
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #Create sobel
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    #Threshold the sobel
    thresh_min = 10
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    #Stack channels
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    #Mask the noise away from the warped image
    vert=np.array([[(150,720),(150, 0), (920,0), (920,720)]], dtype=np.int32)
    binary_warped=region_of_interest(combined_binary,vert)

    #perform histogram for base
    histogram = np.sum(binary_warped[640:,:], axis=0)
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9

    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set the width of the windows +/- margin
    margin = 50

    # Set minimum number of pixels found to recenter window
    minpix = 100

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    #create a blank for the lines to be drawn
    window_img=np.zeros((720,1280,3), np.uint8)

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
     
    #Check for zeros to avoid errors
    if len(lefty)==0 or len(righty)==0:
        return None
    else:
        #Perform poly fit n=2 
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        fit=[]
        for i in left_fit:
            fit.append(i)
        for i in right_fit:
            fit.append(i)

        return fit

def find_lines_2(image_1,fit_find):
    #a function to find the lines if not the first frame, after a histogram has been completed

    #import mtx and dist
    global mtx, dist
    #break out the fit data
    length=len(fit_find)
    length=int(length/2)
    left_fit=fit_find[:length]
    right_fit=fit_find[length:]

    #Undistort image
    img = cv2.undistort(image_1, mtx, dist, None, mtx)

    #create a layover image for later
    layover_image=img

    #warp image
    img=warp(img)

    #Convert to HLS
    hls=cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    
    #go to just s channel
    s_channel = hls[:,:,2]
    
    #threshold s channel 
    s_thresh_min = 170
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    #Create gray image
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #Create sobel
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    #Threshold the sobel
    thresh_min = 20
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    #Stack channels
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    #Mask the noise away
    vert=np.array([[(150,720),(150, 0), (920,0), (920,720)]], dtype=np.int32)
    binary_warped=region_of_interest(combined_binary,vert)

    #perform histogram for base
    histogram = np.sum(binary_warped[640:,:], axis=0)
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9

    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set the width of the windows +/- margin
    margin = 50

    # Set minimum number of pixels found to recenter window
    minpix = 100

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    #create a blank for the lines to be drawn
    window_img=np.zeros((720,1280,3), np.uint8)

    #Non sliding window finding
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 



    #Check for zeros to avoid errors
    if len(lefty)==0 or len(righty)==0:
        return None
    else:
        #Perform poly fit n=2 
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        fit=[]
        for i in left_fit:
            fit.append(i)
        for i in right_fit:
            fit.append(i)

        return fit

def draw_lines(image_1,fit):
    #a function to draw the lines

    #break out the fit data
    length=len(fit)
    length=int(length/2)
    left_fit=fit[:length]
    right_fit=fit[length:]

    #Import and undistor the image
    img = cv2.undistort(image_1, mtx, dist, None, mtx)

    #create a layover image for later
    layover_image=img

    #warp image
    img=warp(img)

    #Write the data to be plotted
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    #create points for cv2.fillpoly
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    #Create a blank image
    window_img=np.zeros((720,1280,3), np.uint8)

    #draw on the image
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,0, 255))
    cv2.polylines(window_img,np.int_([right_line_window1]),False,(255,0,0),10)
    cv2.polylines(window_img,np.int_([right_line_window2]),False,(255,0,0),10)

    #unwarp window image
    window_img=un_warp(window_img)

    #lay window_img on the un modified layover image
    result = cv2.addWeighted(layover_image, 1, window_img, .8, 0)

    #plot the center calc to the result image
    #lane width
    lane_width=right_fitx[0]-left_fitx[0]

    #lane center
    lane_width_2=lane_width/2
    center=left_fitx[0]+lane_width_2

    #distance from center, asumming center of car is center of image
    distance_from_center=640-center

    #Convert distance from pixels to meters
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    #Plot to image
    font = cv2.FONT_HERSHEY_SIMPLEX
    distance_center=str(round(distance_from_center*xm_per_pix,2))
    distance_center="DFC "+distance_center
    cv2.putText(result,distance_center,(30,350),font,4,(255,255,255))

    #Return result 
    return result

#initialize a varible R_last (the value does not matter)
R_last=0
def R_calc(img,fit,t):
    #a function to calculate radius of curvature based on the left (more constant line)

    #create a global varible to be modified
    global R_last

    #Break out left and right
    length=len(fit)
    length=int(length/2)
    left_fit=fit[:length]

    #Write some data
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    
    #Calculate radius of curvature
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    #Convert polynomials to real world dimensions
    #left_fit = np.polyfit(left_fitx*ym_per_pix, ploty*xm_per_pix, 2)
    left_fit=np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)


    y_eval = np.max(ploty)
    R_left = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    
    #% Different
    diff=np.absolute((R_left-R_last)/(R_left+R_last))

    #If the difference is greater than 10% its noisy and use old value, if it is the first frame setup R_last
    if t==0:
        R=R_left
        R_last=R_left
    #elif diff>100.0:
        #xR=R_last
    else:
        R=R_left
        R_last=R_left

    #Put R in the top left corner of the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    R=int(R)
    R=str(R)
    R="Radius "+R
    cv2.putText(img,R,(30,200),font,4,(255,255,255))

    #Return the image
    return img

#Setup the lists
left_fit_0=[]
left_fit_1=[]
left_fit_2=[]
right_fit_0=[]
right_fit_1=[]
right_fit_2=[]

def loop(get_frame,t):
    #a function to apply the functions on the frames of the video

    avg_fit=[]
    global fit_finder
    
    #Get frame
    frame = get_frame(t)

    #Get new_fit
    if t==0:
        new_fit=find_lines(frame)
    else:
        new_fit=find_lines_2(frame,fit_finder)
        
    #Call up global varibles
    global left_fit_0
    global left_fit_1
    global left_fit_2
    global right_fit_0
    global right_fit_1
    global right_fit_2
   
    #Name a list to be used later   
    fit_finder=[]

    #Make sure fit is not none, this would casue and error
    if new_fit != None:
        #Break out new_fit into right and left
        length=len(new_fit)
        length=int(length/2)
        left_fit=new_fit[:length]
        right_fit=new_fit[length:]

        if len(left_fit_0)>0:
            #calculate percent differences only if the len is above zero to mitigate error, also make sure left_fit_0 is not an empty list
            avg_left=(sum(left_fit_0)/len(left_fit_0))
            avg_right=(sum(right_fit_0)/len(right_fit_0))
            diff_left=np.absolute((left_fit[0]-avg_left)/((left_fit[0]+avg_left)/2))
            diff_right=np.absolute((right_fit[0]-avg_right)/((right_fit[0]+avg_right)/2))
        #error threshold     
        error=100    
        #If the length of the runnning average is greater than or equal to zero and less than 10 then add new numbers
        if len(left_fit_0)>=0 and len(left_fit_0)<10:
            left_fit_0.append(left_fit[0])
            left_fit_1.append(left_fit[1])
            left_fit_2.append(left_fit[2])
            right_fit_0.append(right_fit[0])
            right_fit_1.append(right_fit[1])
            right_fit_2.append(right_fit[2])
        #If the length of the running average is target length then delete index 0 and add the new one
        #and if the new value is not greater than 100% of the running average do nothing and use the old running average
        elif diff_left<error and diff_right<error:
            left_fit_0.remove(left_fit_0[0])
            left_fit_1.remove(left_fit_1[0])
            left_fit_2.remove(left_fit_2[0])
            left_fit_0.append(left_fit[0])
            left_fit_1.append(left_fit[1])
            left_fit_2.append(left_fit[2])

            right_fit_0.remove(right_fit_0[0])
            right_fit_1.remove(right_fit_1[0])
            right_fit_2.remove(right_fit_2[0])
            right_fit_0.append(right_fit[0])
            right_fit_1.append(right_fit[1])
            right_fit_2.append(right_fit[2])

    #Take the average and append to avg_fit
    avg_fit.append(sum(left_fit_0)/len(left_fit_0))
    avg_fit.append(sum(left_fit_1)/len(left_fit_1))
    avg_fit.append(sum(left_fit_2)/len(left_fit_2))
    avg_fit.append(sum(right_fit_0)/len(right_fit_0))
    avg_fit.append(sum(right_fit_1)/len(right_fit_1))
    avg_fit.append(sum(right_fit_2)/len(right_fit_2))

    #Create a list of the poly fit coefficents to be fed to find_lines_2 so it has an area to search
    fit_finder.append(left_fit[0])
    fit_finder.append(left_fit[1])
    fit_finder.append(left_fit[2])
    fit_finder.append(right_fit[0])
    fit_finder.append(right_fit[1])
    fit_finder.append(right_fit[2])

    #Draw the lines with avg_fit
    frame=draw_lines(frame,avg_fit)

    #Radius of curvature calc
    frame=R_calc(frame,avg_fit,t)

    #Return the frame with the polygon, lines, R and DFC drawn on it 
    return frame


white_output = 'white123.mp4'
clip1 = VideoFileClip("/Users/johnmcconnell/Desktop/SDC Nano Degree/CarND-Advanced-Lane-Lines-master/project_video.mp4",audio=False)
white_clip = clip1.fl(loop) 
white_clip.write_videofile(white_output, audio=False)











