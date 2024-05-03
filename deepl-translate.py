
"""
Translate from English to Swedish using DeepL UI

Not gonna be efficient or viable, just fun to test it out. 
"""
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import time
import random
from tqdm import tqdm 
import json 
import os
from dotenv import load_dotenv

option = webdriver.ChromeOptions()
#option.add_argument('--no-sandbox')
#option.add_argument('--headless')
#option.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome(options = option)

driver.get('https://www.deepl.com/translator#en/sv/')
# driver.get('https://www.deepl.com/en/login')

def login(username, password):
    time.sleep(1)
    cookies_button = driver.find_element(By.CSS_SELECTOR, 'button.text-deepl-blue')
    cookies_button.click()
    login_button = driver.find_element(By.CSS_SELECTOR, 'button.LoginButton-module--loginButton--d2RRu')
    login_button.click()
    time.sleep(1)
    email_input = driver.find_element(By.ID, 'menu-login-username')
    email_input.send_keys(username)
    password_input = driver.find_element(By.ID, 'menu-login-password')
    password_input.send_keys(password)
    submit_button = driver.find_element(By.ID, 'menu-login-submit')
    submit_button.click()
    time.sleep(1)

def wait_for_text_to_be_present_in_element_value(driver, locator):
    """Wait for the element's value to be non-empty."""
    def _predicate(_driver):
        try:
            # Retrieve the element's value
            element_value = _driver.find_element(*locator).get_attribute('value')
            # Check if the value is not empty
            return element_value != ''
        except:
            # In case element is not found or another error occurs, return False to keep waiting
            return False
    return _predicate

def translate_text(text):
    # Make sure the field is empty
    # Get the text from the input field
    input_field = driver.find_elements(By.CSS_SELECTOR, 'd-textarea.focus-visible-disabled-container')[0]
    input_field.click()
    input_field.send_keys(Keys.CONTROL + "a" + Keys.DELETE)
    input_field.send_keys(text)
    
    # LET IT COOK 
    time.sleep(5)

    # Get the text from the output field
    output_field = driver.find_elements(By.CSS_SELECTOR, 'd-textarea.focus-visible-disabled-container')[1]

    # WebDriverWait(driver, 1000).until(wait_for_text_to_be_present_in_element_value(driver, output_field))
    # Wait for the output field to be populated

    #print(output_field.text)
    return output_field.text

def translate_json(path, output):
    latest_line = 0
    with open(path, 'r') as file:
        lines = file.readlines()

    with open(output, "a") as out: 
        for _, line in enumerate(tqdm(lines[latest_line:], initial=latest_line, total=len(lines))):
            try:
                data = json.loads(line)
                conv_list = data['conversations']
                for conv in conv_list:
                    conv['value'] = translate_text(conv['value'])
            except:
                print("Error")
                continue
            data['conversations'] = conv_list
            json.dump(data, out)
            out.write('\n')
            out.flush()

def translate_for_model(path, output):
    latest_line = 27
    with open(path, 'r') as file:
        lines = file.readlines()

    with open(output, "a") as out: 
        for _, line in enumerate(tqdm(lines[latest_line:], initial=latest_line, total=len(lines))):
            # try:
            data = json.loads(line)
            data['sv'] = translate_text(data['en'])
            # except:
            #     print("Error")
            #     continue
            json.dump(data, out)
            out.write('\n')
            out.flush()

if __name__ == "__main__":
    load_dotenv()
    username = os.environ.get("DEEPL_USERNAME")
    password = os.environ.get("DEEPL_PASSWORD")

    login(username, password)
    translate_json("./bad-examples-en-sv.jsonl", "corrected-examples-en-sv.jsonl")
    
    # longer_string = "The evening light shimmers on the shore\nSoftly the waves echoes around and more \nAs I bask in the sun, my worries are all gone\nThe sound of seagulls I now foolishly ignore \nGlistening sand, beckons me with a silent plea \nGlistening seawater, cool to the touch and refreshingly free \nThe evening brings peace, yet I can't find any \nBut maybe in the morning there'll be time for me\nMy bottled peacefulness, I uncork and pour \nThe sound of the ocean, lulls me even more \nAnd for just a moment I close my eyes and behold \nThe vastness of the ocean, to my soul I now unfold."

    # Generate a string of english words in 5000 characters
    # english_string = generate_english_string(5000)
    # english_string = english_string[:5000]

    # translate_text(english_string)
    # translate_text("The Supreme Court is the highest court in the US.")
    # translate_text(longer_string)

