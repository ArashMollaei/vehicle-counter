import streamlit as st 
import tempfile
import os
import pandas as pd 
import plotly.express as px 
from tracker import VehicleCounter
from config import VEHICLE_CLASSES, DEFAULT_LINE_RATIO
import time


st.set_page_config(page_title= 'Vehicle Counter', layout= 'wide')
st.title('Smart car counter 🚗')
st.markdown('Upload a traffic video and get the number of passes of each type of vehicle along with the tracked video.')


# Sidebar: Settings
st.sidebar.header('Counting line settings ⚙️')
line_ratio = st.sidebar.slider(
    'Line position (height ratio)',
    min_value= 0.5, max_value= 0.95, value= DEFAULT_LINE_RATIO, step= 0.05,
    help= 'The position of the green horizontal line through which the crossing is counted.'
)

model_option = st.sidebar.selectbox(
    'YOLO model',
    ('yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt'),
    index= 0,
    help= 'Larger model, more accurate but slower'
)

# Main part
uploaded_file = st.file_uploader('📁 Select a video', type= ['mp4', 'avi', 'mov', 'mkv'])

def force_remove(filepath, retries=3, delay=0.5):
    for _ in range(retries):
        try:
            os.unlink(filepath)
            return
        except PermissionError:
            time.sleep(delay)

if uploaded_file is not None:
    # Save temporary file
    tfile = tempfile.NamedTemporaryFile(delete= False, suffix= '.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    st.video(video_path)
    st.caption('🎬 Original video')
        
    if st.button('▶️ Start processing', type= 'primary'):
        try:
            with st.spinner('Tracking and counting 🚗 ... This may take a few minutes⏳...'):
                # initialization    
                counter = VehicleCounter(model_name= model_option, line_ratio= line_ratio)
                output_path = 'processed_output.mp4'
                counts, out_path = counter.process_video(video_path, output_path)
                
            st.success('✅ Processing completed successfully.')
            
            
            # Show statistics
            st.subheader('Counting results 📊')
            if counts:
                df = pd.DataFrame([
                    {'type': VEHICLE_CLASSES.get(cls, str(cls)), 'number': count}
                    for cls, count in  counts.items()
                ])
                fig = px.bar(df, x= 'type', y= 'number', color= 'type', text_auto= True,
                            title= 'Number of passes of each type of vehicle')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info('No vehicles were identified in this video.')
            
            # Download button
            with open(out_path, 'rb') as f:
                st.download_button(
                    label= '📥 Download the final video',
                    data= f,
                    file_name= 'traffic_processed.mp4',
                    mime= 'video/mp4'
                )
                
        except Exception as e:
            st.error(f'❌ An error occurred: {e}')
        finally:
            # Clear temporary file
            if os.path.exists(video_path):
                force_remove(video_path)