import torch
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig
import streamlit as st
import os
from io import StringIO
from PIL import Image, ImageEnhance
import docx2txt
import pdfplumber
import cv2
import pytesseract
import os
import re


process_dir = os.getcwd()

def load_image(image_file):
    img = Image.open(image_file)
    return img
    

def ocr_extraction(image):
    
    try:       
        enhance_img = Image.open(image)
        enhancer = ImageEnhance.Contrast(enhance_img)
        factor = 1.3 #increase contrast
        im_output = enhancer.enhance(factor)
        improved_image = os.path.join(process_dir, "test.jpg")
        im_output.save(improved_image)
                
        img = cv2.imread(improved_image)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    except:
        print("Error found for image")
        
         
    try:  
        ocr_result_eng_raw = pytesseract.image_to_string(img_gray, lang="eng", config="--psm 4")
    except:
        print("Error found while extracting ocr")
    
    try:    
        ocr_result_eng = ocr_result_eng_raw.split('\n')
        
        my_new_list = []
        for line in ocr_result_eng:
            
            if not re.match(r'^\s*$', line):
                cleaned_line = re.findall('[a-zA-Z0-9/d+/s+/ ]', line)
 
                if cleaned_line:
                    my_new_list.append(cleaned_line)
        
        extracted_text = []
        for i in my_new_list:
            s = ''.join(i)
            extracted_text.append(s)

    except:
        print("Error found while parsing and formatting ocr extracted test")

    return extracted_text


def text_summarize_BART(text):
    
    tokenizer=BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    
    inputs = tokenizer.batch_encode_plus([text],return_tensors='pt', max_length=1024)
    summary_ids = model.generate(inputs['input_ids'], early_stopping=True)
    
    # Decoding the summary
    bart_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return bart_summary



# def text_summarize_BART_file(file):
    
#     with open(file, "r", encoding='utf8') as f:
#         text = f.read().replace("\n", "")
#         text = text.replace("\t", "")
#         text = text.replace("\s", "")
    
#     tokenizer=BartTokenizer.from_pretrained('facebook/bart-large-cnn')
#     model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    
#     inputs = tokenizer.batch_encode_plus([text],return_tensors='pt')
#     summary_ids = model.generate(inputs['input_ids'], early_stopping=True, max_length=1024)
    
#     # Decoding the summary
#     bart_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
#     return bart_summary


def main():
    st.title("Text Summarization")
    html_temp = """
    <div style="background-color:green;padding:10px">
    <h2 style="color:white;text-align:center;">Text Summarization ML App</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    text_data = st.text_input("Text","Type Here")
    
    result=""
    if st.button("Summarize"):
        
        result = text_summarize_BART(text_data)
        st.success('Summarized Text: {}'.format(result))
     
    filename = st.file_uploader("Upload File", type=["txt","jpg","pdf","docx"])
#    print(filename)
    if ((filename is not None) and ('txt' in str(filename))):
        text_data = str(filename.read(), "utf-8", errors='ignore')
        
        text_data = text_data.replace("\n", "")
        text_data = text_data.replace("\t", "")
        text_data = text_data.replace("\s", "")
        
        st.write(text_data)
    
    elif ((filename is not None) and ('jpg' in str(filename))):
        st.image(load_image(filename))
        text_data = ocr_extraction(filename)
        text_data = str(text_data)
        
   
    elif ((filename is not None) and ('docx' in str(filename))):
        text_data = docx2txt.process(filename)
        st.write(text_data)
        
    elif ((filename is not None) and ('pdf' in str(filename))):
        
        with pdfplumber.open(filename) as pdf:
            pages = pdf.pages[0]
            text_data = str(pages.extract_text())
            st.write(text_data)
 
    else:
        pass
    
    result=""
    if st.button("Summarize Text"):
        result = text_summarize_BART(text_data)
        st.success('Summarized Text {}'.format(result))
    

# img1 = "global-warming-effects.jpg"
# ext_text = ocr_extraction(img1)
# ext_text = str(ext_text)




if __name__=='__main__':
    main()

