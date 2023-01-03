import logging

logging.basicConfig(level=logging.WARNING,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M',
    handlers=[logging.FileHandler('thesis.log','a','utf-8'),])

def logging_func(num,input_video,frame_index):
    
    if num ==1:
        print("lack of right or left lanes to calculate vanishing_point")
        logging.warning('Name {},stop in frame{},Reason: {}'.format(input_video,str(frame_index),"lack of right or left lanes to calculate vanishing_point"))
                
