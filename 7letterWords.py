import requests
import zipfile
import io
import os

# Step 1: Download and extract the Turkish word database
url = "https://github.com/ekartal/turkce-kelime-database/archive/master.zip"
response = requests.get(url)

# Extract the content of the zip file
with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    z.extractall("turkce_kelime_database")

# Step 2: Process the files and filter the words
seven_letter_filtered_words = []

# Directory where the word files are stored
directory = "turkce_kelime_database/turkce-kelime-database-master"

# Read and process each word file
for file_name in os.listdir(directory):
    if file_name.endswith(".txt"):
        with open(os.path.join(directory, file_name), 'r', encoding='utf-8') as file:
            words = file.read().splitlines()
            # Filter for 7-letter words with the condition: 2nd and 7th letters same, others different
            for word in words:
                if len(word) == 7 and word[1] == word[6] and len(set(word[:6] + word[2:6])) == 6:
                    seven_letter_filtered_words.append(word)

# Step 3: Sort the filtered words alphabetically
sorted_words = sorted(seven_letter_filtered_words)

# Step 4: Save the sorted words to a file
with open("sorted_7_letter_words.txt", "w", encoding="utf-8") as f:
    for word in sorted_words:
        f.write(word + "\n")

# Output confirmation
print("Words saved to 'sorted_7_letter_words.txt'")