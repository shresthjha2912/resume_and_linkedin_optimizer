import os
import re  # Add this line
import time
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
from groq import Groq
import warnings
import random
import urllib.parse  # Also add this if it's missing
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Load API key with proper error handling
try:
    # Try to get from Streamlit secrets first
    groq_api_key = st.secrets["GROQ_API_KEY"]
except (FileNotFoundError, KeyError):
    # Fall back to environment variable
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ API key not found in Streamlit secrets or environment variables. Please set it up.")

client = Groq(api_key=groq_api_key) if groq_api_key else None

# Helper: add vertical space
def add_vertical_space(lines=1):
    for _ in range(lines):
        st.write("")

def streamlit_config():
    st.set_page_config(page_title='Resume Analyzer AI', layout="wide")
    st.markdown('<h1 style="text-align: center;">Resume Analyzer AI</h1>', unsafe_allow_html=True)

class resume_analyzer:

    @staticmethod
    def pdf_to_chunks(pdf):
        pdf_reader = PdfReader(pdf)
        # Convert None returns from extract_text() to empty strings
        text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200, length_function=len)
        return text_splitter.split_text(text=text)

    @staticmethod
    def llama_api_request(prompt):
        try:
            completion = client.chat.completions.create(
                model="deepseek-r1-distill-llama-70b",
                messages=[{"role": "user", "content": prompt}],
                temperature=1,
                max_tokens=1024,
                top_p=1,
                stream=False,
                stop=None,
            )
            return completion.choices[0].message.content
        except Exception as e:
            return str(e)

    @staticmethod
    def process_resume(pdf, query_template):
        if pdf is not None:
            try:
                with st.spinner('Processing...'):
                    pdf_chunks = resume_analyzer.pdf_to_chunks(pdf)
                    prompt_text = query_template("\n".join(pdf_chunks))
                    response = resume_analyzer.llama_api_request(prompt_text)
                return response
            except Exception as e:
                return str(e)
        return "Please upload your resume."

    @staticmethod
    def resume_summary():
        with st.form(key='Summary'):
            add_vertical_space(1)
            pdf = st.file_uploader(label='Upload Your Resume', type='pdf')
            submit = st.form_submit_button(label='Submit')

        if submit:
            summary = resume_analyzer.process_resume(pdf, lambda text: f"Summarize the following resume:\n{text}")
            st.markdown('<h4 style="color: orange;">Summary:</h4>', unsafe_allow_html=True)
            st.write(summary)

    @staticmethod
    def resume_strength():
        with st.form(key='Strength'):
            pdf = st.file_uploader(label='Upload Your Resume', type='pdf')
            submit = st.form_submit_button(label='Submit')

        if submit:
            strength = resume_analyzer.process_resume(pdf, lambda text: f"Analyze the strengths in the resume:\n{text}")
            st.markdown('<h4 style="color: orange;">Strength:</h4>', unsafe_allow_html=True)
            st.write(strength)

    @staticmethod
    def resume_weakness():
        with st.form(key='Weakness'):
            pdf = st.file_uploader(label='Upload Your Resume', type='pdf')
            submit = st.form_submit_button(label='Submit')

        if submit:
            weakness = resume_analyzer.process_resume(pdf, lambda text: f"Analyze the weaknesses and suggest improvements for this resume:\n{text}")
            st.markdown('<h4 style="color: orange;">Weakness and Suggestions:</h4>', unsafe_allow_html=True)
            st.write(weakness)

    @staticmethod
    def job_title_suggestion():
        with st.form(key='Job Titles'):
            pdf = st.file_uploader(label='Upload Your Resume', type='pdf')
            submit = st.form_submit_button(label='Submit')

        if submit:
            job_titles = resume_analyzer.process_resume(pdf, lambda text: f"Suggest relevant job titles based on this resume:\n{text}")
            st.markdown('<h4 style="color: orange;">Job Titles:</h4>', unsafe_allow_html=True)
            st.write(job_titles)

    @staticmethod
    def auto_job_finder():
        add_vertical_space(2)
        with st.form(key='auto_job_finder'):
            add_vertical_space(1)
            col1, col2, col3 = st.columns([0.5, 0.3, 0.2], gap='medium')
            with col1:
                pdf = st.file_uploader(label='Upload Your Resume', type='pdf')
            with col2:
                job_location = st.text_input(label='Job Location', value='India')
            with col3:
                job_count = st.number_input(label='Job Count', min_value=1, value=5, step=1)

            # Submit Button
            add_vertical_space(1)
            submit = st.form_submit_button(label='Find Matching Jobs')
            add_vertical_space(1)
        
        debug_mode = st.checkbox("Debug Mode", value=True)
        
        add_vertical_space(2)
        
        if submit:
            if pdf is not None and job_location != '':
                driver = None
                try:
                    # Extract job titles from resume
                    with st.spinner('Analyzing resume to find ideal job titles...'):
                        prompt = "Based on this resume, what are the 3 most relevant job titles this person should apply for? Return ONLY a comma-separated list of job titles with no additional text or formatting."
                        response = resume_analyzer.process_resume(pdf, lambda text: prompt + f"\n{text}")
                        
                        # Clean up the response to extract only the job titles
                        job_title_input = clean_ai_response(response)
                        
                        # Add fallback titles if nothing valid was found
                        if len(job_title_input) < 2:
                            job_title_input = ["Software Engineer", "Data Scientist", "Developer"]
                        
                        st.markdown('<h4 style="color: orange;">Suggested Job Titles:</h4>', unsafe_allow_html=True)
                        st.write(", ".join(job_title_input))
                    
                    # Continue with LinkedIn scraping using these job titles
                    with st.spinner('Chrome Webdriver Setup Initializing...'):
                        driver = linkedin_scraper.webdriver_setup()
                    
                    with st.spinner('Loading Job Listings...'):
                        link = linkedin_scraper.build_url(job_title_input, job_location)
                        
                        if debug_mode:
                            st.write(f"Job titles to search: {job_title_input}")
                            st.write(f"Search URL: {link}")

                        # Continue with the rest of the code...
                
                except Exception as e:
                    add_vertical_space(2)
                    st.markdown(f'<h5 style="text-align: center;color: orange;">Error searching for jobs: {e}</h5>', unsafe_allow_html=True)
                
                finally:
                    if driver:
                        driver.quit()
            
            # If User Click Submit Button and No Resume
            elif pdf is None:
                st.markdown(f'<h5 style="text-align: center;color: orange;">Please upload your resume</h5>', 
                            unsafe_allow_html=True)
            
            elif job_location == '':
                st.markdown(f'<h5 style="text-align: center;color: orange;">Job Location is Empty</h5>', 
                            unsafe_allow_html=True)

class linkedin_scraper:

    @staticmethod
    def webdriver_setup():
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        # Add user agent to appear more like a real browser
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0'
        ]
        options.add_argument(f'user-agent={random.choice(user_agents)}')
        
        # Disable automation flags
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option('excludeSwitches', ['enable-automation'])
        options.add_experimental_option('useAutomationExtension', False)

        driver = webdriver.Chrome(options=options)
        driver.maximize_window()
        
        # Mask WebDriver to avoid detection
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        return driver

    @staticmethod
    def get_userinput():
        add_vertical_space(2)
        with st.form(key='linkedin_scrap'):
            add_vertical_space(1)
            col1, col2, col3 = st.columns([0.5, 0.3, 0.2], gap='medium')
            with col1:
                job_title_input = st.text_input(label='Job Title')
                job_title_input = job_title_input.split(',')
            with col2:
                job_location = st.text_input(label='Job Location', value='India')
            with col3:
                job_count = st.number_input(label='Job Count', min_value=1, value=1, step=1)

            # Submit Button
            add_vertical_space(1)
            submit = st.form_submit_button(label='Submit')
            add_vertical_space(1)
        
        return job_title_input, job_location, job_count, submit

    @staticmethod
    def build_url(job_titles, location):
        # Create a proper search keyword from job titles
        keywords = " OR ".join([f'"{title}"' for title in job_titles])
        
        # URL encode the components
        encoded_keywords = urllib.parse.quote(keywords)
        encoded_location = urllib.parse.quote(location)
        
        # Build LinkedIn search URL with time filter for recent jobs (last week)
        url = f"https://www.linkedin.com/jobs/search/?keywords={encoded_keywords}&location={encoded_location}&f_TPR=r604800"
        
        return url

    @staticmethod
    def open_link(driver, link, debug_mode=False):
        max_attempts = 3
        attempt = 0
        
        while attempt < max_attempts:
            try:
                driver.get(link)
                # Add random wait time to mimic human behavior
                time.sleep(random.uniform(2, 5))
                
                # Check for various elements that would indicate page load
                selectors = [
                    'span.switcher-tabs__placeholder-text.m-auto',
                    'div.jobs-search-results-list',
                    'h1.jobs-search-results-list__title'
                ]
                
                for selector in selectors:
                    try:
                        driver.find_element(by=By.CSS_SELECTOR, value=selector)
                        if debug_mode:
                            st.write(f"Page loaded successfully. Found element: {selector}")
                        return True
                    except NoSuchElementException:
                        continue
                
                # Check if we're facing a CAPTCHA or login wall
                if "Sign in" in driver.title or "Verify" in driver.title:
                    if debug_mode:
                        st.warning("LinkedIn may be requiring sign-in or CAPTCHA verification")
                    
                attempt += 1
                if debug_mode:
                    st.write(f"Attempt {attempt}: Page not fully loaded, retrying...")
                time.sleep(3)  # Wait before retry
                
            except Exception as e:
                if debug_mode:
                    st.error(f"Error loading page: {str(e)}")
                attempt += 1
                time.sleep(3)  # Wait before retry
        
        if debug_mode:
            st.error("Failed to load LinkedIn page after multiple attempts")
        return False

    @staticmethod
    def link_open_scrolldown(driver, link, job_count, debug_mode=False):
        # Open the Link in LinkedIn
        success = linkedin_scraper.open_link(driver, link, debug_mode)
        
        if not success:
            # Try an alternative URL
            if debug_mode:
                st.write("Trying alternative LinkedIn URL format...")
            alt_link = link.replace("www.linkedin.com", "in.linkedin.com")
            success = linkedin_scraper.open_link(driver, alt_link, debug_mode)
            
            if not success:
                if debug_mode:
                    st.error("Could not access LinkedIn jobs search.")
                return False
        
        # Take a screenshot for debugging if needed
        if debug_mode:
            try:
                screenshot_path = "linkedin_page.png"
                driver.save_screenshot(screenshot_path)
                st.image(screenshot_path, caption="LinkedIn Jobs Page Screenshot")
            except:
                st.write("Could not capture screenshot")
        
        # Scroll Down the Page
        scroll_pause_time = random.uniform(0.7, 1.5)  # Random pause to mimic human behavior
        
        for i in range(0, max(3, job_count)):
            # Handle sign-in popups
            try:
                # Various selectors for close buttons on modals
                dismiss_selectors = [
                    "button[data-tracking-control-name='public_jobs_contextual-sign-in-modal_modal_dismiss']>icon>svg",
                    "button.modal__dismiss",
                    "button.artdeco-modal__dismiss"
                ]
                
                for selector in dismiss_selectors:
                    try:
                        close_buttons = driver.find_elements(by=By.CSS_SELECTOR, value=selector)
                        if close_buttons:
                            close_buttons[0].click()
                            time.sleep(0.5)
                            if debug_mode:
                                st.write("Closed a popup")
                            break
                    except:
                        continue
            except:
                pass
            
            # Scroll down the Page to End
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(scroll_pause_time)
            
            # Simulate some random human-like scrolling
            for _ in range(2):
                scroll_amount = random.randint(300, 700)
                driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
                time.sleep(random.uniform(0.3, 0.8))
            
            # Click on See More Jobs Button if Present using multiple possible selectors
            see_more_selectors = [
                "button[aria-label='See more jobs']",
                "button.infinite-scroller__show-more-button",
                "button.see-more-jobs"
            ]
            
            for selector in see_more_selectors:
                try:
                    see_more_buttons = driver.find_elements(by=By.CSS_SELECTOR, value=selector)
                    if see_more_buttons:
                        see_more_buttons[0].click()
                        time.sleep(2)
                        if debug_mode:
                            st.write("Clicked 'See more jobs' button")
                        break
                except:
                    continue
        
        return True

    @staticmethod
    def job_title_filter(scrap_job_title, user_job_title_input, debug_mode=False):
        if not scrap_job_title or scrap_job_title.strip() == "":
            return np.nan
            
        # User Job Title Convert into Lower Case
        user_input = [i.lower().strip() for i in user_job_title_input if i.strip()]
        
        # scraped Job Title Convert into Lower Case
        scrap_title = scrap_job_title.lower().strip()
        
        if debug_mode:
            st.write(f"Comparing '{scrap_title}' with user inputs: {user_input}")
        
        # More lenient matching - check if ANY significant words from user input appear
        # in the scraped title
        for title in user_input:
            # Split the user input title into individual words
            words = title.split()
            
            # Check for exact title match
            if title in scrap_title:
                if debug_mode:
                    st.write(f"Exact title match found: {title}")
                return scrap_job_title
                
            # Check for significant words (3+ characters)
            matched_words = 0
            for word in words:
                if len(word) >= 3 and word in scrap_title:
                    matched_words += 1
                    
            # If at least half of significant words match, consider it a match
            if matched_words >= max(1, len(words) // 2):
                if debug_mode:
                    st.write(f"Partial match found with {matched_words} words")
                return scrap_job_title
            
        return np.nan

    @staticmethod
    def scrap_company_data(driver, job_title_input, job_location, debug_mode=False):
        # Multiple selectors to try for different LinkedIn page structures
        company_selectors = [
            'h4[class="base-search-card__subtitle"]',
            'h4[class*="base-search-card__subtitle"]',
            'span[class*="company-name"]',
            'div[class*="company-name"]',
            'a[class*="job-card-container__company-name"]'
        ]
        
        location_selectors = [
            'span[class="job-search-card__location"]',
            'span[class*="job-search-card__location"]',
            'div[class*="job-card-container__metadata"] span',
            'span[class*="location"]'
        ]
        
        title_selectors = [
            'h3[class="base-search-card__title"]',
            'h3[class*="base-search-card__title"]',
            'h3[class*="job-card-list__title"]',
            'a[class*="job-card-container__link"]'
        ]
        
        # Try each selector until we find elements
        company_name = []
        for selector in company_selectors:
            company = driver.find_elements(by=By.CSS_SELECTOR, value=selector)
            if company:
                company_name = [i.text for i in company if i.text.strip()]
                if debug_mode:
                    st.write(f"Found {len(company_name)} companies using selector: {selector}")
                break
        
        company_location = []
        for selector in location_selectors:
            location = driver.find_elements(by=By.CSS_SELECTOR, value=selector)
            if location:
                company_location = [i.text for i in location if i.text.strip()]
                if debug_mode:
                    st.write(f"Found {len(company_location)} locations using selector: {selector}")
                break
        
        job_title = []
        for selector in title_selectors:
            title = driver.find_elements(by=By.CSS_SELECTOR, value=selector)
            if title:
                job_title = [i.text for i in title if i.text.strip()]
                if debug_mode:
                    st.write(f"Found {len(job_title)} job titles using selector: {selector}")
                break
        
        # For URLs, check multiple XPath patterns
        url_patterns = [
            '//a[contains(@href, "/jobs/view/")]',
            '//a[contains(@href, "/jobs/")]',
            '//div[contains(@class, "job-card")]//a'
        ]
        
        website_url = []
        for pattern in url_patterns:
            url = driver.find_elements(by=By.XPATH, value=pattern)
            if url:
                website_url = [i.get_attribute('href') for i in url if i.get_attribute('href')]
                if debug_mode:
                    st.write(f"Found {len(website_url)} URLs using pattern: {pattern}")
                break
        
        # If we still don't have data, try one more approach:
        if not company_name or not job_title:
            if debug_mode:
                st.write("Trying alternative approach to find job listings...")
            
            job_cards = driver.find_elements(by=By.CSS_SELECTOR, value='div.job-card-container')
            if job_cards:
                if debug_mode:
                    st.write(f"Found {len(job_cards)} job cards")
                
                # Extract data from individual job cards
                for card in job_cards:
                    try:
                        # Title
                        title_elem = card.find_element(by=By.CSS_SELECTOR, value='h3')
                        if title_elem and title_elem.text.strip():
                            job_title.append(title_elem.text.strip())
                        
                        # Company
                        company_elem = card.find_element(by=By.CSS_SELECTOR, value='h4')
                        if company_elem and company_elem.text.strip():
                            company_name.append(company_elem.text.strip())
                        
                        # Location
                        location_elem = card.find_element(by=By.CSS_SELECTOR, value='span[class*="location"]')
                        if location_elem and location_elem.text.strip():
                            company_location.append(location_elem.text.strip())
                        
                        # URL
                        url_elem = card.find_element(by=By.TAG_NAME, value='a')
                        if url_elem and url_elem.get_attribute('href'):
                            website_url.append(url_elem.get_attribute('href'))
                    except:
                        continue
        
        if debug_mode:
            st.write(f"Data collected: {len(company_name)} companies, {len(job_title)} titles, {len(company_location)} locations, {len(website_url)} URLs")
        
        # If still no data or inconsistent lengths, provide dummy data for testing
        if debug_mode and (not company_name or not job_title or not company_location or not website_url):
            st.warning("No job listings found or inconsistent data. LinkedIn might be blocking scraping.")
            
            # For testing, create some dummy data if in debug mode
            if debug_mode:
                st.write("Creating placeholder data for testing...")
                dummy_size = 3
                company_name = ["Company " + str(i) for i in range(1, dummy_size+1)]
                job_title = [f"{title.strip()} Position" for title in job_title_input[:dummy_size]]
                if not job_title:
                    job_title = ["Software Developer", "Data Analyst", "Project Manager"][:dummy_size]
                company_location = [job_location] * dummy_size
                website_url = ["https://linkedin.com/jobs/view/1"] * dummy_size
        
        # Find the minimum length among all lists
        if company_name and job_title and company_location and website_url:
            min_length = min(len(company_name), len(company_location), len(job_title), len(website_url))
            
            # Truncate lists to ensure consistent length
            company_name = company_name[:min_length]
            company_location = company_location[:min_length]
            job_title = job_title[:min_length]
            website_url = website_url[:min_length]
            
            # Create DataFrame using truncated lists
            data = {
                'Company Name': company_name,
                'Job Title': job_title,
                'Location': company_location,
                'Website URL': website_url
            }
            df = pd.DataFrame(data)
            
            # Apply filters with debug flag
            df['Job Title'] = df['Job Title'].apply(lambda x: linkedin_scraper.job_title_filter(x, job_title_input, debug_mode))
            
            # More lenient location filter (partial match)
            df['Location'] = df['Location'].apply(lambda x: x if job_location.lower() in x.lower() or any(word in x.lower() for word in job_location.lower().split()) else np.nan)
            
            # Drop Null Values and Reset Index
            df = df.dropna()
            df.reset_index(drop=True, inplace=True)
            
            if debug_mode:
                st.write(f"After filtering: {len(df)} matching jobs found")
                if len(df) == 0:
                    st.write("No matches after filtering. Job titles found but filtered out:")
                    st.write(job_title)
            
            return df
        else:
            if debug_mode:
                st.warning("Could not collect complete job data")
            return pd.DataFrame()

    @staticmethod
    def scrap_job_description(driver, df, job_count, debug_mode=False):
        if len(df) == 0:
            if debug_mode:
                st.warning("No job listings to scrape descriptions from")
            return df
        
        # Get URL into List
        website_url = df['Website URL'].tolist()
        
        # Scrap the Job Description
        job_description = []
        description_count = 0
        
        description_selectors = [
            'div[class="show-more-less-html__markup relative overflow-hidden"]',
            'div[class*="show-more-less-html__markup"]',
            'div[class*="description__text"]',
            'section[class*="description"]'
        ]
        
        show_more_selectors = [
            'button[data-tracking-control-name="public_jobs_show-more-html-btn"]',
            'button[class*="show-more-less-button"]',
            'button[aria-label="Show more, visually expands the job description"]'
        ]

        for i in range(0, min(len(website_url), job_count)):
            try:
                # Open the job listing page
                if debug_mode:
                    st.write(f"Opening job listing {i+1}/{min(len(website_url), job_count)}")
                
                linkedin_scraper.open_link(driver, website_url[i], debug_mode)
                time.sleep(random.uniform(1, 2))  # Random wait
                
                # Try to click "Show More" button using different selectors
                show_more_clicked = False
                for selector in show_more_selectors:
                    try:
                        show_more_buttons = driver.find_elements(by=By.CSS_SELECTOR, value=selector)
                        if show_more_buttons:
                            show_more_buttons[0].click()
                            time.sleep(1)
                            show_more_clicked = True
                            if debug_mode:
                                st.write("Clicked 'Show more' button")
                            break
                    except:
                        continue
                
                # Try different selectors to get job description
                description_text = None
                for selector in description_selectors:
                    try:
                        description_elements = driver.find_elements(by=By.CSS_SELECTOR, value=selector)
                        if description_elements:
                            description_text = description_elements[0].text
                            if description_text.strip():
                                break
                    except:
                        continue
                
                # If still no description, try a more general approach
                if not description_text:
                    try:
                        # Look for any large text block that might be the description
                        main_content = driver.find_element(by=By.CSS_SELECTOR, value='main')
                        paragraphs = main_content.find_elements(by=By.TAG_NAME, value='p')
                        if paragraphs:
                            description_text = "\n".join([p.text for p in paragraphs if p.text.strip()])
                    except:
                        pass
                
                # Add description if found
                if description_text and len(description_text.strip()) > 0 and description_text not in job_description:
                    job_description.append(description_text)
                    description_count += 1
                    if debug_mode:
                        st.write(f"Successfully extracted job description ({len(description_text)} characters)")
                else:
                    placeholder = "Description Not Available"
                    if debug_mode:
                        st.warning(f"Could not extract job description for job {i+1}")
                        placeholder = f"Description Not Available (URL: {website_url[i]})"
                    job_description.append(placeholder)
            
            except Exception as e:
                if debug_mode:
                    st.error(f"Error extracting job description: {str(e)}")
                job_description.append('Description Not Available')
            
            # Add some random delay between requests
            time.sleep(random.uniform(1, 3))
        
        # Filter to match the number of descriptions we managed to get
        df = df.iloc[:len(job_description), :]
        
        # Add Job Description in Dataframe
        df['Job Description'] = pd.DataFrame(job_description, columns=['Description'])
        
        # If in debug mode, keep all entries for inspection
        if not debug_mode:
            df['Job Description'] = df['Job Description'].apply(lambda x: np.nan if x=='Description Not Available' else x)
            df = df.dropna()
            df.reset_index(drop=True, inplace=True)
        
        return df

    @staticmethod
    def display_data_userinterface(df_final):
        # Display the Data in User Interface
        add_vertical_space(1)
        if len(df_final) > 0:
            for i in range(0, len(df_final)):
                st.markdown(f'<h3 style="color: orange;">Job Posting Details : {i+1}</h3>', unsafe_allow_html=True)
                st.write(f"Company Name : {df_final.iloc[i,0]}")
                st.write(f"Job Title    : {df_final.iloc[i,1]}")
                st.write(f"Location     : {df_final.iloc[i,2]}")
                st.write(f"Website URL  : {df_final.iloc[i,3]}")

                with st.expander(label='Job Description'):
                    st.write(df_final.iloc[i, 4])
                add_vertical_space(3)
        else:
            st.markdown(f'<h5 style="text-align: center;color: orange;">No Matching Jobs Found</h5>', 
                                unsafe_allow_html=True)
            st.markdown('<p style="text-align: center;">LinkedIn may be blocking automated access or no matching jobs were found. Try:</p>', unsafe_allow_html=True)
            st.markdown('<ul><li>Using broader job titles</li><li>Checking a different location</li><li>Running in debug mode to see more details</li></ul>', unsafe_allow_html=True)

    @staticmethod
    def main():
        # Initially set driver to None
        driver = None
        
        try:
            # Add new UI elements for resume upload and analysis
            st.markdown('<h3 style="text-align: center;">LinkedIn Job Search</h3>', unsafe_allow_html=True)
            tab1, tab2 = st.tabs(["Resume Analysis", "Manual Search"])
            
            with tab1:
                with st.form(key='resume_job_search'):
                    add_vertical_space(1)
                    col1, col2, col3 = st.columns([0.5, 0.3, 0.2], gap='medium')
                    with col1:
                        pdf = st.file_uploader(label='Upload Your Resume', type='pdf')
                    with col2:
                        job_location = st.text_input(label='Job Location', value='India')
                    with col3:
                        job_count = st.number_input(label='Job Count', min_value=1, value=5, step=1)

                    # Submit Button
                    add_vertical_space(1)
                    submit_resume = st.form_submit_button(label='Find Matching Jobs')
                
                # Move debug mode outside the form
                debug_mode = st.checkbox("Debug Mode", value=False, 
                                    help="Shows detailed information about the scraping process")
                
                if submit_resume:
                    if pdf is not None and job_location != '':
                        # Extract job titles from resume
                        with st.spinner('Analyzing resume to find ideal job titles...'):
                            prompt = "Based on this resume, what are the 3 most relevant job titles this person should apply for? Return ONLY a comma-separated list of job titles with no additional text or formatting."
                            response = resume_analyzer.process_resume(pdf, lambda text: prompt + f"\n{text}")
                            
                            # Clean up the response
                            job_title_input = clean_ai_response(response)
                            
                            # Add fallback titles if needed
                            if len(job_title_input) < 2:
                                job_title_input = ["Software Engineer", "Data Scientist", "Developer"]
                            
                            st.markdown('<h4 style="color: orange;">Suggested Job Titles:</h4>', unsafe_allow_html=True)
                            st.write(", ".join(job_title_input))
                        
                        # Continue with LinkedIn scraping
                        with st.spinner('Chrome Webdriver Setup Initializing...'):
                            driver = linkedin_scraper.webdriver_setup()
                        
                        with st.spinner('Loading Job Listings...'):
                            link = linkedin_scraper.build_url(job_title_input, job_location)
                            
                            if debug_mode:
                                st.write(f"Job titles to search: {job_title_input}")
                                st.write(f"Search URL: {link}")

                            # Open the Link in LinkedIn and Scroll Down the Page
                            success = linkedin_scraper.link_open_scrolldown(driver, link, job_count, debug_mode)
                            
                            if not success and debug_mode:
                                st.error("Failed to properly load LinkedIn jobs page")

                        with st.spinner('Scraping Job Details...'):
                            # Scraping the Company Name, Location, Job Title and URL Data
                            df = linkedin_scraper.scrap_company_data(driver, job_title_input, job_location, debug_mode)
                            
                            if debug_mode and len(df) > 0:
                                st.write("Initial jobs found:")
                                st.write(df[['Company Name', 'Job Title', 'Location']])
                            
                            # Scraping the Job Description Data
                            df_final = linkedin_scraper.scrap_job_description(driver, df, job_count, debug_mode)
                        
                        # Display the Data in User Interface
                        linkedin_scraper.display_data_userinterface(df_final)
                    
                    elif pdf is None:
                        st.markdown('<h5 style="text-align: center;color: orange;">Please upload your resume</h5>', 
                                    unsafe_allow_html=True)
                    
                    elif job_location == '':
                        st.markdown('<h5 style="text-align: center;color: orange;">Job Location is Empty</h5>', 
                                    unsafe_allow_html=True)
            
            with tab2:
                # Original manual job search form
                with st.form(key='linkedin_manual_search'):
                    add_vertical_space(1)
                    col1, col2, col3 = st.columns([0.5, 0.3, 0.2], gap='medium')
                    with col1:
                        job_title_input = st.text_input(label='Job Title')
                        job_title_input = job_title_input.split(',')
                    with col2:
                        job_location = st.text_input(label='Job Location', value='India')
                    with col3:
                        job_count = st.number_input(label='Job Count', min_value=1, value=5, step=1)

                    # Submit Button
                    add_vertical_space(1)
                    submit = st.form_submit_button(label='Search')
                
                # Move debug mode outside the form
                if not 'debug_mode' in locals():  # Only define if not already defined
                    debug_mode = st.checkbox("Debug Mode", value=False, 
                                        help="Shows detailed information about the scraping process")
                
                if submit:
                    if job_title_input != [''] and job_location != '':
                        with st.spinner('Chrome Webdriver Setup Initializing...'):
                            driver = linkedin_scraper.webdriver_setup()
                                        
                        with st.spinner('Loading Job Listings...'):
                            # build URL based on User Job Title Input
                            link = linkedin_scraper.build_url(job_title_input, job_location)
                            
                            if debug_mode:
                                st.write(f"Job titles to search: {job_title_input}")
                                st.write(f"Search URL: {link}")

                            # Open the Link in LinkedIn and Scroll Down the Page
                            success = linkedin_scraper.link_open_scrolldown(driver, link, job_count, debug_mode)
                            
                            if not success and debug_mode:
                                st.error("Failed to properly load LinkedIn jobs page")

                        with st.spinner('Scraping Job Details...'):
                            # Scraping the Company Name, Location, Job Title and URL Data
                            df = linkedin_scraper.scrap_company_data(driver, job_title_input, job_location, debug_mode)
                            
                            if debug_mode and len(df) > 0:
                                st.write("Initial jobs found:")
                                st.write(df[['Company Name', 'Job Title', 'Location']])
                            
                            # Scraping the Job Description Data
                            df_final = linkedin_scraper.scrap_job_description(driver, df, job_count, debug_mode)
                        
                        # Display the Data in User Interface
                        linkedin_scraper.display_data_userinterface(df_final)
                    
                    # If User Click Submit Button and Job Title is Empty
                    elif job_title_input == ['']:
                        st.markdown(f'<h5 style="text-align: center;color: orange;">Job Title is Empty</h5>', 
                                    unsafe_allow_html=True)
                    
                    elif job_location == '':
                        st.markdown(f'<h5 style="text-align: center;color: orange;">Job Location is Empty</h5>', 
                                    unsafe_allow_html=True)

        except Exception as e:
            add_vertical_space(2)
            st.markdown(f'<h5 style="text-align: center;color: orange;">Error: {e}</h5>', unsafe_allow_html=True)

        finally:
            # Close the browser driver if it was created
            if driver:
                driver.quit()

def clean_ai_response(response):
    """Clean the AI response to extract just the job titles"""
    # Remove thinking tags if present
    cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    
    # Further clean up any other noise
    cleaned = cleaned.replace('\n', ' ').strip()
    
    # Extract job titles
    job_titles = [title.strip() for title in cleaned.split(',') if title.strip()]
    
    return job_titles

# Main application logic
if __name__ == "__main__":
    streamlit_config()
    
    # Create horizontal menu with just two options
    selected = option_menu(
        menu_title=None,
        options=["Resume Analysis", "LinkedIn Job Search"],
        icons=["file-earmark-text", "linkedin"],
        default_index=0,
        orientation="horizontal",
    )
    
    if selected == "Resume Analysis":
        # Navigation for Resume Analysis
        analysis_option = st.sidebar.radio("Choose Analysis", ["Summary", "Strengths", "Weaknesses", "Job Titles"])
        
        if analysis_option == "Summary":
            resume_analyzer.resume_summary()
        elif analysis_option == "Strengths":
            resume_analyzer.resume_strength()
        elif analysis_option == "Weaknesses":
            resume_analyzer.resume_weakness()
        elif analysis_option == "Job Titles":
            resume_analyzer.job_title_suggestion()
            
    elif selected == "LinkedIn Job Search":
        linkedin_scraper.main()