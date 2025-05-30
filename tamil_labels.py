"""
Tamil labels mapping for sign language gestures.
This file contains the mapping between English category names and Tamil text.
Update this mapping based on your specific dataset categories.
"""

def get_tamil_labels():
    """
    Returns a dictionary mapping English category names to Tamil text.
    
    Since the actual dataset categories are not known, this provides
    a template that should be updated based on the actual TSL dataset.
    """
    
    # Actual TSL Dataset categories mapping
    tamil_labels = {
        # Your TSL dataset categories with Tamil labels
        'அழுகை': 'அழுகை',           # Crying
        'சாப்பிடு': 'சாப்பிடு',       # Eat
        'பானம் குடித்தல்': 'பானம் குடித்தல்',  # Drinking
        'வணக்கம்': 'வணக்கம்',       # Hello/Greeting
        'வரவேற்பு': 'வரவேற்பு',     # Welcome
    }
    
    return tamil_labels

def add_tamil_label(english_label, tamil_text):
    """
    Add a new Tamil label mapping.
    This function can be used to dynamically add new mappings.
    """
    tamil_labels = get_tamil_labels()
    tamil_labels[english_label] = tamil_text
    return tamil_labels

def get_all_categories():
    """
    Get all available categories in both English and Tamil.
    """
    tamil_labels = get_tamil_labels()
    categories = []
    
    for english, tamil in tamil_labels.items():
        categories.append({
            'english': english,
            'tamil': tamil
        })
    
    return categories

def validate_tamil_text(text):
    """
    Basic validation for Tamil text.
    Checks if the text contains Tamil Unicode characters.
    """
    tamil_unicode_range = range(0x0B80, 0x0BFF + 1)
    
    for char in text:
        if ord(char) in tamil_unicode_range:
            return True
    
    return False

# Instructions for updating this file:
"""
To customize this file for your specific TSL dataset:

1. Open your TSL Dataset folder and note the exact folder names for the 5 categories
2. Update the tamil_labels dictionary with the correct mappings:
   - Replace the keys (left side) with your actual folder names
   - Replace the values (right side) with the corresponding Tamil text

Example:
If your folders are named:
- 'namaste'
- 'dhanyawad'
- 'paani'
- 'ghar'
- 'vyakti'

Update the dictionary like this:
tamil_labels = {
    'namaste': 'வணக்கம்',
    'dhanyawad': 'நன்றி',
    'paani': 'தண்ணீர்',
    'ghar': 'வீடு',
    'vyakti': 'நபர்',
}

3. Make sure the Tamil text is correctly encoded in Unicode
4. Test the mappings by running the application
"""
