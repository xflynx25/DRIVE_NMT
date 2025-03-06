# Simple Google Cloud Translation API wrapper
import os
import requests
from dotenv import load_dotenv
from typing import List, Union, Dict, Any
import pandas as pd
import time
from tqdm import tqdm  # For progress bars
import random

# Load environment variables from .env file
load_dotenv()

# Get API key from .env file
API_KEY = os.environ.get("GoogleCloudAPIKEY")

def robust_translate_request(
    text_list: List[str],
    target_language: str,
    source_language: str,
    max_retries: int = 5,
    initial_backoff: float = 1.0,
    max_backoff: float = 32.0
) -> Union[List[str], None]:
    """Make a robust translation request with exponential backoff for retries.
    
    Args:
        text_list: List of texts to translate
        target_language: Target language code
        source_language: Source language code
        max_retries: Maximum number of retry attempts
        initial_backoff: Initial backoff time in seconds
        max_backoff: Maximum backoff time in seconds
        
    Returns:
        List of translated texts or None if all retries fail
    """
    # API endpoint for Google Translate
    url = "https://translation.googleapis.com/language/translate/v2"
    
    # Request parameters
    params = {
        'key': API_KEY
    }
    
    # Request data (using data instead of params for POST body)
    data = {
        'q': text_list,
        'target': target_language,
        'source': source_language,
        'format': 'text'
    }
    
    # Try the request with exponential backoff
    backoff_time = initial_backoff
    
    for attempt in range(max_retries):
        try:
            # Make the API request
            response = requests.post(url, params=params, json=data)
            
            # Check if the request was successful
            if response.status_code == 200:
                # Parse the JSON response
                result = response.json()
                
                # Extract and return the translated texts
                return [t['translatedText'] for t in result['data']['translations']]
            
            # Handle specific error codes
            elif response.status_code == 411:  # Length Required
                print(f"Error 411: Content-length issue. Reducing batch size and retrying...")
                
                # If we have multiple texts, try with a smaller batch
                if len(text_list) > 1:
                    # Split the batch in half
                    mid = len(text_list) // 2
                    first_half = text_list[:mid]
                    second_half = text_list[mid:]
                    
                    # Process each half
                    print(f"Splitting batch of {len(text_list)} into {len(first_half)} and {len(second_half)}")
                    
                    # Process first half
                    first_results = robust_translate_request(
                        first_half, target_language, source_language, 
                        max_retries, initial_backoff, max_backoff
                    )
                    
                    # Add a delay between requests
                    time.sleep(2)
                    
                    # Process second half
                    second_results = robust_translate_request(
                        second_half, target_language, source_language, 
                        max_retries, initial_backoff, max_backoff
                    )
                    
                    # Combine results if both were successful
                    if first_results is not None and second_results is not None:
                        return first_results + second_results
                    else:
                        return None
                else:
                    # If we only have one text and still get this error, it might be too long
                    print(f"Single text may be too long ({len(text_list[0])} chars). Handling as special case...")
                    # Here you could implement special handling for very long texts
                    # For now, we'll just continue with the retry logic
            
            elif response.status_code == 429:  # Too Many Requests
                print(f"Error 429: Rate limit exceeded. Will retry after backoff...")
            
            else:
                print(f"Error {response.status_code}: {response.text}")
            
            # If we get here, the request failed and we should retry
            print(f"Attempt {attempt+1}/{max_retries} failed. Retrying in {backoff_time:.1f} seconds...")
            
        except Exception as e:
            print(f"Request exception: {e}")
            print(f"Attempt {attempt+1}/{max_retries} failed. Retrying in {backoff_time:.1f} seconds...")
        
        # Wait before retrying with exponential backoff and jitter
        jitter = random.uniform(0, 0.5 * backoff_time)  # Add randomness to avoid thundering herd
        time.sleep(backoff_time + jitter)
        
        # Increase backoff time for next attempt, but don't exceed max_backoff
        backoff_time = min(backoff_time * 2, max_backoff)
    
    # If we get here, all retries failed
    print(f"All {max_retries} attempts failed. Giving up on this batch.")
    return None

def translate_sentences(
    sentences: Union[str, List[str]],
    target_language: str = "fr",
    source_language: str = "en"
) -> Union[str, List[str]]:
    """Translate a list of sentences using Google Cloud Translation API.
    
    Args:
        sentences: A single string or list of strings to translate
        target_language: The target language code (e.g., 'fr' for French)
        source_language: The source language code (e.g., 'en' for English)
        
    Returns:
        A single translated string or list of translated strings
    """
    # Check if API key is set
    if not API_KEY:
        print("WARNING: API key is not set. Please check your .env file.")
        print("Make sure you have a variable named 'GoogleCloudAPIKEY' set.")
        return None
    
    # Convert single string to list for consistent handling
    is_single_string = isinstance(sentences, str)
    text_list = [sentences] if is_single_string else sentences
    
    # Use the robust request function
    translations = robust_translate_request(text_list, target_language, source_language)
    
    # Return the appropriate format based on input
    if translations is None:
        return None
    elif is_single_string:
        return translations[0]
    else:
        return translations

# Simple example usage
if __name__ == "__main__":

    TRANSLATE_TYPE = 3

    # Maximum batch size for API calls (adjust based on API limits)
    BATCH_SIZE = 100 # Reduced batch size for more reliable processing

    if TRANSLATE_TYPE == 0:
        # Example 1: Translate a single sentence
        sentence = "Hello, how are you today?"
        translation = translate_sentences(sentence, source_language="en", target_language="dz")
        print(f"Original: {sentence}")
        print(f"Translation: {translation}")
        print()
    
    
    elif TRANSLATE_TYPE == 1:   
    # Example 2: Translate multiple sentences
        sentences = [
            "The weather is beautiful outside.",
            "I would like to learn more about Bhutan.",
            "Neural machine translation is an exciting field of research."
        ]
        translations = translate_sentences(sentences, target_language="dz")
        print(sentences, '\n\n', translations)
    
    
    elif TRANSLATE_TYPE == 2:
        # Example 3: Translate a file
        file_path = "./Scraping/example_sentences.txt"
        sentences = open(file_path, 'r').readlines()
        translations = translate_sentences(sentences, target_language="dz")
        for original, translated in zip(sentences, translations):
            print(f"Original: {original}",end='')
            print(f"Translation: {translated}\n")
        print()


    elif TRANSLATE_TYPE == 3:
        # Example 4: Translate a csv
        file_path = "./Data/Prepared/OpenLanguageData_Flores-plus_Parallel_Test_Dataset.csv"
        
        # Read the CSV file
        print(f"Reading CSV file: {file_path}")
        try:
            df = pd.read_csv(file_path)
            total_rows = df.shape[0]
            print(f"CSV shape: {df.shape} ({total_rows} rows)")
            
            
            # Uncomment one of these lines:
            df_sample = df.head(10).copy()  # Take only the first 10 rows for testing
            df_sample = df.copy()  # Process the full dataset
            
            sample_size = len(df_sample)
            print(f"Processing {sample_size} rows for translation")
            
            # Create new columns for translations
            df_sample['eng_to_dzo'] = None  # English to Dzongkha translation
            df_sample['dzo_to_eng'] = None  # Dzongkha to English translation
            
            # Optimized batch processing approach
            print("\n1. Processing all English to Dzongkha translations...")
            
            # Process in batches to avoid API limits
            for i in tqdm(range(0, sample_size, BATCH_SIZE)):
                batch_end = min(i + BATCH_SIZE, sample_size)
                batch_df = df_sample.iloc[i:batch_end]
                
                # Get all English texts for this batch
                eng_texts = batch_df['eng'].tolist()
                
                print(f"Batch {i//BATCH_SIZE + 1}: Processing {len(eng_texts)} English texts...")
                
                # Translate all English texts to Dzongkha at once
                eng_to_dzo_translations = translate_sentences(eng_texts, 
                                                             source_language="en", 
                                                             target_language="dz")
                
                # Check if translation was successful
                if eng_to_dzo_translations is None:
                    print(f"Warning: Translation failed for batch {i//BATCH_SIZE + 1}. Skipping...")
                    continue
                
                # Update the DataFrame with translations
                for j, idx in enumerate(batch_df.index):
                    df_sample.at[idx, 'eng_to_dzo'] = eng_to_dzo_translations[j]
                
                # Add a small delay between batches to avoid hitting API rate limits
                if batch_end < sample_size:
                    time.sleep(3)  # Increased delay between batches
            
            # Save intermediate results
            intermediate_file = "./Data/Prepared/translation_evaluation_GoogleTranslate_intermediate.csv"
            df_sample.to_csv(intermediate_file, index=False)
            print(f"Intermediate results saved to: {intermediate_file}")
            
            print("\n2. Processing all Dzongkha to English translations...")
            
            # Process in batches
            for i in tqdm(range(0, sample_size, BATCH_SIZE)):
                batch_end = min(i + BATCH_SIZE, sample_size)
                batch_df = df_sample.iloc[i:batch_end]
                
                # Get all Dzongkha texts for this batch
                dzo_texts = batch_df['dzo'].tolist()
                
                print(f"Batch {i//BATCH_SIZE + 1}: Processing {len(dzo_texts)} Dzongkha texts...")
                
                # Translate all Dzongkha texts to English at once
                dzo_to_eng_translations = translate_sentences(dzo_texts, 
                                                             source_language="dz", 
                                                             target_language="en")
                
                # Check if translation was successful
                if dzo_to_eng_translations is None:
                    print(f"Warning: Translation failed for batch {i//BATCH_SIZE + 1}. Skipping...")
                    continue
                
                # Update the DataFrame with translations
                for j, idx in enumerate(batch_df.index):
                    df_sample.at[idx, 'dzo_to_eng'] = dzo_to_eng_translations[j]
                
                # Add a small delay between batches
                if batch_end < sample_size:
                    time.sleep(.5)  # Increased delay between batches
            
            # Save the results to a new CSV file
            output_file = "./Data/Prepared/translation_evaluation_GoogleTranslate.csv"
            df_sample.to_csv(output_file, index=False)
            print(f"\nTranslations completed and saved to: {output_file}")
            
            # Display a sample of the results
            print("\nSample of translations:")
            sample_display = min(3, sample_size)
            for idx, row in df_sample.head(sample_display).iterrows():
                print(f"\nRow {idx+1}/{sample_size}:")
                print(f"Original English: {row['eng'][:50]}...")
                print(f"Original Dzongkha: {row['dzo'][:50]}...")
                print(f"English→Dzongkha: {row['eng_to_dzo'][:50]}...")
                print(f"Dzongkha→English: {row['dzo_to_eng'][:50]}...")
            
            print(f"\nTotal processing time for {sample_size} rows completed successfully.")

            os.remove(intermediate_file)
            
        except Exception as e:
            print(f"Error processing CSV: {e}")
            import traceback
            traceback.print_exc()