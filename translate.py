from googletrans import Translator
import pandas as pd

# Function to translate text from Vietnamese to English
def translate_vietnamese_to_english(vietnamese_text):
    try:
        translator = Translator()
        translation = translator.translate(vietnamese_text, src='vi', dest='en')
        return translation.text
    except Exception as e:
        print(f"Translation error: {e}")
        return vietnamese_text  # Keep the original text if translation fails

# Read the CSV file and assign it to the ticket_data DataFrame
ticket_data = pd.read_csv("to_translate.csv")

num = 0
for i, vietnamese_text in enumerate(ticket_data.iloc[:, 12]):  
    english_translation = translate_vietnamese_to_english(vietnamese_text)
    ticket_data.iloc[i, 12] = english_translation  # Update the DataFrame with the translation
    num += 1
    print(f'Đã dịch tới bản thứ {num}')

# Save the updated DataFrame to a new CSV file named "new.csv"
ticket_data.to_csv("translated.csv", index=False)

# text = 'con mẹ mày'
# text_trans = translate_vietnamese_to_english(text)
# print(text_trans)