
"""
Translate from English to Swedish using DeepL UI
"""
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import random
import os
from dotenv import load_dotenv

option = webdriver.ChromeOptions()
driver = webdriver.Chrome(options = option)

driver.get('https://www.deepl.com/translator#en/sv/')

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
    # Get the text from the input field
    input_field = driver.find_elements(By.CSS_SELECTOR, 'd-textarea.focus-visible-disabled-container')[0]

    input_field.click()
    input_field.send_keys(text)
    
    # LET IT COOK 
    # time.sleep(1)

    # Get the text from the output field
    output_field = driver.find_elements(By.CSS_SELECTOR, 'd-textarea.focus-visible-disabled-container')[1]

    # WebDriverWait(driver, 1000).until(wait_for_text_to_be_present_in_element_value(driver, output_field))
    # Wait for the output field to be populated



    print(output_field.text)

def generate_english_string(length):
    words = ['apple', 'banana', 'car', 'dog', 'elephant', 'flower', 'guitar', 'house', 'internet', 'jungle']
    english_string = ' '.join(random.choices(words, k=length))
    return english_string

if __name__ == "__main__":
    load_dotenv()
    username = os.environ.get("USERNAME")
    password = os.environ.get("PASSWORD")

    login(username, password)


    # Generate a string of english words in 5000 characters
    # english_string = generate_english_string(5000)
    # english_string = english_string[:5000]

    # translate_text(english_string)
    translate_text("The Supreme Court is the highest court in the US.")

