import base64
import requests
import os
from talon import Module, actions, app, screen
from PIL import Image, ImageDraw, ImageFont

mod = Module()

# Configure Gemini API
with open('gemini_api_key', 'r') as f:
    API_KEY = f.read().strip()

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-thinking-exp-01-21:generateContent"

# Main prompt for the Gemini API
GEMINI_PROMPT = """
You are a highly precise computer vision assistant. Your task is to identify a UI element based on user voice command and output the exact pixel coordinates to complete the following command:
"{phrase}"

Think step by step: multiple short thoughts allowed per section.
Use extremely concise language. Each thought max 5-7 words.

Section 1 - Command Interpretation:
Background:
- The text given comes from speech-to-text software, which may parse the audio incorrectly
- The text may even be not meant for you. (For example, overhearing background noises) In this case, there is nothing for you to do. Just output "None"
Output:
- Fix any speech recognition errors
- If you believe the message is not a command meant for you, output "None" and end the output. In this case, there is no need to produce sections 2 and 3
- Identify which specific UI element the user wants to click
- What is the approximate location of this with respect to other UI elements?
- Add more thoughts if needed

Section 2 - Grid Analysis:
Image has a coordinate grid:
- Each red square is 50x50 pixels
- Grid boxes are labeled
- Grey lines every 25 pixels between main colored lines
  - Top of box is y=0
  - Bottom of box is y=50
  - Left edge is x=0
  - Right edge is x=50
Output:
- Restate the last line of Section 1 word for word
- Note the ID of the bounding box containing the UI element
- Use grey lines to determine the quadrant
- If needed, refine even further for precise position (more specific than 25 pixels)
- Calculate center point of target element
- The final coordinates should be **within** the box (between 0 and 50)

Section 3 - Final Output:
Output:
- [The box label]
- [The calculated coordinates]

Rules:
- Final output format of Section 3 must be exactly: "ID\\nx,y"
- The second to last line contains ONLY the box ID. No explanation or other text.
- The final line contains ONLY the coordinates. No labels, no explanation.
""".strip()

GREY = '#808080'
RED = '#FF0000'  # Color for all grid lines

def to_base26(num, upper_case=False):
    """Convert a number to base 26 using letters.
    
    Args:
        num: Number to convert
        upper_case: If True, use uppercase A-Z. If False, use lowercase a-z.
    """
    if num < 0:
        return None
    
    if num == 0:
        return 'A' if upper_case else 'a'
        
    base = ord('A') if upper_case else ord('a')
        
    digits = []
    while num > 0:
        remainder = num % 26
        digits.append(chr(base + remainder))
        num //= 26
    
    return ''.join(reversed(digits))

def coords_to_id(coords):
    """Convert (x, y) coordinates to a box ID string.
    
    Args:
        coords: Tuple of (x, y) coordinates
        
    Returns:
        String ID in format 'xyY' where x is lowercase a-z (can be multiple letters)
        and Y is uppercase A-Z
    """
    if not isinstance(coords, (tuple, list)) or len(coords) != 2:
        return None
        
    x, y = coords
    if x < 0 or y < 0:
        return None
        
    return f"{to_base26(x)}{to_base26(y, upper_case=True)}"

def id_to_coords(box_id):
    """Convert a box ID string to (x, y) coordinates.
    
    Args:
        box_id: String ID in format 'xyY' where x is base 26 lowercase a-z (can be multiple letters)
        and Y is base 26 uppercase A-Z
        
    Returns:
        Tuple of (x, y) coordinates or None if invalid
    """
    if len(box_id) < 2:
        return None
        
    x = 0
    y = 0
    for c in box_id:
      if c.islower():
        x = x * 26 + (ord(c) - ord('a'))
      else:
        y = y * 26 + (ord(c) - ord('A'))
    
    return (x, y)

def draw_grid(image_path):
    """Draw a 50px grid with labels on the image."""
    # Open image
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    # Get dimensions
    width, height = img.size
    
    # Draw grey lines first (every 25px, excluding 50px positions)
    for x in range(25, width, 25):
        if x % 50 != 0:  # Skip positions where main grid lines will be
            draw.line([(x, 0), (x, height)], fill=GREY, width=1)
    for y in range(25, height, 25):
        if y % 50 != 0:  # Skip positions where main grid lines will be
            draw.line([(0, y), (width, y)], fill=GREY, width=1)
    
    # Draw vertical colored lines
    for x in range(0, width, 50):
        draw.line([(x, 0), (x, height)], fill=RED, width=1)
    
    # Draw horizontal colored lines
    for y in range(0, height, 50):
        draw.line([(0, y), (width, y)], fill=RED, width=1)
    
    # Add coordinates at lattice points with larger font size
    try:
        # Try to load a system font with larger size
        font = ImageFont.truetype("arial.ttf", 17)
    except:
        # Fallback to default font if custom font fails
        font = ImageFont.load_default()
    
    for x in range(0, width, 50):
        for y in range(0, height, 50):
            # Create the coordinate text using coords_to_id
            text = coords_to_id((x//50, y//50))
            # Get text dimensions
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            # Calculate centered position
            text_x = x + 25 - text_width/2
            text_y = y + 25 - text_height/2
            # Draw centered text
            draw.text((text_x, text_y), text, fill=RED, font=font)
    
    # Save the modified image
    img.save(image_path)

@mod.capture(rule="({user.vocabulary} | <phrase>)+")
def phrase(m) -> str:
    """Capture a phrase."""
    return str(m)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def ensure_temp_dir():
    """Ensure temporary directory exists."""
    temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir

def capture_and_process_screenshot():
    """Capture screen, add grid, and return encoded image data."""
    screenshot = screen.capture_rect(screen.main_screen().rect)
    temp_dir = ensure_temp_dir()
    temp_file = os.path.join(temp_dir, 'temp_screenshot.png')
    screenshot.write_file(temp_file)
    draw_grid(temp_file)
    return encode_image(temp_file)

def prepare_api_request(phrase: str, image_data: str):
    """Prepare the API request payload."""
    return {
        "contents": [{
            "parts": [
                {"text": GEMINI_PROMPT.format(phrase=phrase)},
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": image_data
                    }
                }
            ]
        }]
    }

def handle_api_response(response_json):
    """Extract coordinates from API response."""
    if 'candidates' not in response_json or not response_json['candidates']:
        raise ValueError("No response from Gemini")
    
    text = response_json['candidates'][0]['content']['parts'][0]['text'].strip()
    print(GEMINI_PROMPT)
    print(f"Received Gemini response:\n```\n{text}\n```")
    lines = text.split('\n')
    return lines[-2].strip(), lines[-1].strip()

def process_coordinates(box_id: str, coordinates: str):
    """Process coordinate string into mouse movement and click."""
    print(f'{box_id=}')
    # Convert box_id to coordinates
    box_coords = id_to_coords(box_id)
    print(box_coords)
    if box_coords is None:
        raise ValueError("Invalid box ID format")
    
    box_x, box_y = box_coords
    x, y = map(float, coordinates.split(','))
    
    actions.mouse_move(box_x * 50 + x, box_y * 50 + y)
    actions.mouse_click(0)  # 0 = left click

@mod.action_class
class Actions:
    def process_magic_command(phrase: str):
        """Process a magic command phrase."""
        temp_file = os.path.join(ensure_temp_dir(), 'temp_screenshot.png')
        try:
            # Capture and process screenshot
            image_data = capture_and_process_screenshot()
            print('asdf')
            
            # Prepare and send API request
            payload = prepare_api_request(phrase, image_data)
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": API_KEY
            }
            
            print(f'Received "magic" command: {phrase}')
            
            try:
                response = requests.post(API_URL, headers=headers, json=payload)
                response.raise_for_status()
                
                # Handle response
                box_id, coordinates = handle_api_response(response.json())
                
                try:
                    process_coordinates(box_id, coordinates)
                except ValueError:
                    app.notify(body=f"Invalid coordinates format", title='Gemini')
                    
            except requests.exceptions.RequestException as e:
                app.notify(body=f"API Error: {str(e)}", title='Gemini')
            except ValueError as e:
                app.notify(body=str(e), title='Gemini')
                
        except Exception as e:
            app.notify(body=f"Screenshot Error: {str(e)}", title='Gemini')
            if os.path.exists(temp_file):
                os.remove(temp_file)