import re
from datetime import date

def detect_birthday(text):
    """
    to find and parse a birthday from user input.
    """
    
    # for format mm-dd-yy and mm/dd/yy
    # \b for word boundary
    # \d{1,2} for one or 2 digit
    # [-/\s] for dash or slash or space
    # \d{2} for two digits
    # For example: 05-15-99 or 12/31/00
    pattern = r'\b(\d{1,2})[-/\s](\d{1,2})[-/\s](\d{2})\b'
    match = re.search(pattern, text)
    
    if match:
        # get month, day, year
        month = int(match.group(1))
        day = int(match.group(2))
        year = int(match.group(3))
        # now find the century
        if year <= 26:
            year = 2000 + year
        else:
            year = 1900 + year
        # Try to create a date object - this will fail if the date is invalid
        return date(year, month, day)
    
    # for  dd-mm-yyyy or dd/mm/yyyy or dd mm yyyy
    pattern = r'\b(\d{1,2})[-/\s](\d{1,2})[-/\s](\d{4})\b'
    match = re.search(pattern, text)
    
    if match:
        # get day, month, year
        day = int(match.group(1))
        month = int(match.group(2))
        year = int(match.group(3))
        return date(year, month, day)
    
    # for formats like "15 January 1999" or "25 Dec 2000"
    months_full = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
        'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12
    }
    
    # Short month names (3 letters)
    months_short = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    
    # match regex for this format
    pattern = r'\b(\d{1,2})\s+([a-zA-Z]+)\s+(\d{4})\b'
    match = re.search(pattern, text)
    
    if match:
        day = int(match.group(1))
        month_name = match.group(2).lower() # lower case for matching
        year = int(match.group(3))
        
        # find month val from dict
        if month_name in months_full:
            month = months_full[month_name]
        elif month_name in months_short:
            month = months_short[month_name]
        else:
            return None
        
        # return date
        return date(year, month, day)
    return None

def calculate_age(birthday):
    """
    find age from input birthdate
    """
    today = date.today()
    age = today.year - birthday.year
    
    # now check if birthday has passed this year
    if today.month < birthday.month:
        age -= 1
    elif today.month == birthday.month and today.day < birthday.day:
        age -= 1

    return age

def detect_mood(text):
    """
    find the mood of user
    """
    text_lower = text.lower()
    
    # for happy mood
    happy_patterns = r'\b(happy|happi|hapi|hapy|glad|cheerful|joyful|great|good|fine|excellent|wonderful|amazing)\b'
    if re.search(happy_patterns, text_lower):
        return 'happy'
    
    # for sad mood
    sad_patterns = r'\b(sad|sead|sadd|unhappy|down|depressed|low|blue|gloomy|miserable)\b'    
    if re.search(sad_patterns, text_lower):
        return 'sad'
    
    # for angry mood
    angry_patterns = r'\b(angry|angri|mad|furious|annoyed|irritated|upset|frustrated)\b'    
    if re.search(angry_patterns, text_lower):
        return 'angry'
    
    # for excited mood
    excited_patterns = r'\b(excited|excitd|thrilled|pumped|eager|enthusiastic)\b'
    if re.search(excited_patterns, text_lower):
        return 'excited'
    
    # for tired/exhausted mood
    tired_patterns = r'\b(tired|tird|exhausted|sleepy|fatigued|weary|drained)\b'
    if re.search(tired_patterns, text_lower):
        return 'tired'
    
    # for neutral/okay mood
    neutral_patterns = r'\b(okay|ok|fine|alright|neutral|normal|meh)\b'
    if re.search(neutral_patterns, text_lower):
            return 'neutral'

    return None

def mood_response(mood):
    """
    response based on mood
    """
    responses = {
        'happy': "Great to hear that you're happy!",
        'sad': "Sorry to hear that you're feeling sad.",
        'angry': "You look angry, calm down.",
        'excited': "Your excitement is contagious.",
        'tired': "You should get some rest as you are tired.",
        'neutral': "You seem to be feeling neutral."
    }
    return responses.get(mood, "I see. Thank you for sharing.")

def extract_surname(full_name):
    """
    Get the surname from fullname
    """
    
    # split the name into parts
    parts = re.sub(r'\s+', ' ', full_name.strip()).split()
    
    # for different len of split
    if len(parts) == 0:
        return None
    elif len(parts) == 1:
        return parts[0]
    else:
        return parts[-1]

def main():
    """
    Main chatbot function
    """
    print("chatbot: Hi there! I'm reggy chatbot")
    print("chatbot: What is your full name?")
    
    # Get user name
    name_input = input("You: ").strip()
    
    # get surname
    surname = extract_surname(name_input)
    
    if surname:
        print(f"chatbot: Nice to meet you, {name_input}! Your surname is {surname}.")
    else:
        print(f"chatbot: Nice to meet you, {name_input}!")
    
    # ask for birthday
    print("chatbot: When is your birthday?")
    
    birthday_input = input("You: ").strip()
    birthday = detect_birthday(birthday_input)
    
    if birthday:
        # get age
        age = calculate_age(birthday)
        print(f"chatbot: So your birthday is {birthday.strftime('%B %d, %Y')}.")
        print(f"chatbot: You are {age} years old!")
    else:
        print("chatbot: I could not understand your birthdate.")
    
    # ask for user feeling
    print("chatbot: How are you feeling today?")
    
    mood_input = input("You: ").strip()
    mood = detect_mood(mood_input)
    
    if mood:
        print(f"chatbot: {mood_response(mood)}")
    else:
        print("chatbot: I am not sure how you are feeling.")
    
    print("chatbot: bye!")

if __name__ == "__main__":
    main()
