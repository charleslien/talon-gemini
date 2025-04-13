import io
import tempfile

import google.generativeai as genai
from google.generativeai.types import content_types

from talon import Context, Module, actions, app, clip, screen

GEMINI_MODEL = 'gemini-1.5-flash'
mod = Module()
with open('gemini_api_key', 'r') as f:
    gemini_api_key = f.read()
genai.configure(api_key=gemini_api_key.strip())


@mod.capture(rule="({user.vocabulary} | <phrase>)+")
def phrase(m) -> str:
    return apply_formatting(m)


@mod.action_class
class Actions:

    def process_phrase(phrase: str):
        """Processes the captured phrase."""
        screenshot = screen.capture_rect(screen.main_screen().rect)
        clip.set_image(screenshot)

        with tempfile.NamedTemporaryFile(delete=True, suffix='.png') as f:
            screenshot.write_file(f.name)
            uploaded = genai.upload_file(f.name)
        history.append(
            content_types.to_content({
                'parts':
                [f'Speech-to-text: {phrase}\n', 'Current screen:', uploaded],
                'role':
                'user'
            }))
        response = model.generate_content(history)
        history.append(response.candidates[0].content)
        for part in response.candidates[0].content.parts:
            if 'function_call' in part:
                fn_name = part.function_call.name
                named_args = part.function_call.args
                tools_by_name[fn_name](**named_args)
            if 'text' in part:
                app.notify(body=part.text, title='Gemini')


_TYPE_ID = {str: 1, float: 2, int: 3, bool: 4, list: 5, dict: 6}
llm_tools = []
tools_by_name = {}


def get_function_declaration_proto(fn,
                                   params) -> genai.protos.FunctionDeclaration:
    return genai.protos.FunctionDeclaration(
        name=fn.__name__,
        description=fn.__doc__,
        parameters=genai.protos.Schema(**params))


def register(params):

    def decorator(fn):
        llm_tools.append(get_function_declaration_proto(fn, params))
        tools_by_name[fn.__name__] = fn
        return fn

    return decorator


@register(
    params={
        'type_': _TYPE_ID[dict],
        'properties': {
            'x': {
                'type_': _TYPE_ID[float]
            },
            'y': {
                'type_': _TYPE_ID[float]
            }
        },
        'required': ('x', 'y')
    })
def mouse_move(x: float, y: float) -> None:
    '''Moves the mouse to the given coordinates.

    (x=0, y=0) is the top left corner of the screen.

    If the size of the screen is 1920x1200:
        (x=810, y=600): approximately the middle of the screen
        (x=810, y=0): the center top of the screen
        (x=1920, y=1200): the bottom right corner of the screen

    If coordinates given are negative or bigger than the size of the screen, they will be clamped to 0 and the size of the screen. e.g. x=-100 will be converted to x=0.
    '''
    actions.mouse_move(x, y)


@register(
    params={
        'type_': _TYPE_ID[dict],
        'properties': {
            'button': {
                'type_': _TYPE_ID[str],
                'enum': ['LEFT', 'RIGHT', 'MIDDLE']
            },
            'press_action': {
                'type_': _TYPE_ID[str],
                'enum': ['CLICK', 'HOLD', 'RELEASE']
            }
        },
        'required': ('button', 'press_action')
    })
def mouse_button(button: str, press_action: str) -> None:
    '''Presses the given mouse button without moving the mouse.'''
    button_id = {'LEFT': 0, 'RIGHT': 1, 'MIDDLE': 2}[button]
    press_action = {
        'CLICK': actions.mouse_click,
        'HOLD': actions.mouse_drag,
        'RELEASE': actions.mouse_release
    }
    press_action(button_id)


@register(
    params={
        'type_': _TYPE_ID[dict],
        'properties': {
            'y': {
                'type_':
                _TYPE_ID[float],
                'description':
                'Positive values scroll down, negative values scroll up. Scrolls that many pixels. If not provided, does not scroll vertically.'
            },
            'x': {
                'type_':
                _TYPE_ID[float],
                'description':
                'Positive values scroll right, negative values scroll left. Scrolls that many pixels. If not provided, does not scroll horizontally.'
            },
            'by_lines': {
                'type_': _TYPE_ID[bool],
                'description': 'Scroll lines instead of pixels.'
            }
        },
    })
def mouse_scroll(y: float = 0, x: float = 0, by_lines: bool = False):
    '''Scroll using the mouse wheel.'''
    mouse_scroll(y=y, x=x, by_lines=by_lines)


@register(
    params={
        'type_': _TYPE_ID[dict],
        'properties': {
            'text': {
                'type_': _TYPE_ID[str],
            }
        },
        'required': ('text', )
    })
def text_insert(text: str) -> None:
    '''Types the given text.'''
    actions.insert(text)


@register(
    params={
        'type_': _TYPE_ID[dict],
        'properties': {
            'keys': {
                'type_': _TYPE_ID[str],
            },
            'platform': {
                'type_': _TYPE_ID[str],
                'enum': [app.platform],
            }
        },
        'required': ('keys', 'platform')
    })
def key_press(keys: str, platform: str) -> None:
    '''Press one or more keys by name, space-separated.

    Available keys:
        a z 0 9 - + ( ) etc.
        alt super ctrl shift cmd
        left right up down
        backspace bksp
        delete del
        escape esc
        pgup pageup pgdown pagedown
        return enter
        tab space
        home end
        ralt rctrl rshift
        capslock scroll_lock insert
        f1 f2 ... f35
        mute voldown volup play stop play_pause prev next rewind fast_forward
        menu help sysreq printscr compose
        brightness_up brightness_down
        backlight_up backlight_down backlight_toggle
        keypad_0 keypad_1 ... keypad_9
        keypad_clear keypad_enter keypad_separator keypad_decimal keypad_plus
        keypad_multiply keypad_divide keypad_minus keypad_equals

    Keys can be held down with key:down and released with key:up
    Example: key_press("shift:down", ...) or key_press("shift:up", ...)

    Modification keys can be added attached with dashes (`-`).
    Example:
        key_press("cmd-q", "mac"): quits the current application.
        keypress("super", "linux"): opens the Activities Overview or Application Launcher.
        key_press("ctrl-shift-t", "windows"): reopens last tab on most browsers.
    '''
    actions.key(keys)


@register(
    params={
        'type_': _TYPE_ID[dict],
        'properties': {
            'duration': {
                'type_': _TYPE_ID[float],
            }
        },
        'required': ('duration', )
    })
def sleep_seconds(duration: float) -> None:
    '''Waits for the given amount of time (in seconds).'''
    actions.sleep(duration)


model = genai.GenerativeModel(GEMINI_MODEL, tools=llm_tools)
history = content_types.to_contents([
    {
        'parts': [
            'Hello, my name is Gemini, your personal voice-powered assistant!\n',
            'How can I help you today?'
        ],
        'role':
        'model',
    },
    {
        'parts': [
            "Hi Gemini, I'll be using speech-to-text engine to communicate with you as part of an accessibility app.\n",
            'You will have access to a screenshot of my screen, and I may refer to UI elements currently on the screen.\n',
            'You can control my computer, using mouse movements and keyboard inputs to complete the task I give you, much like a normal person would use a computer.\n',
            "The software I'm using might cut me off early, so if you think I have not given you a complete command, please wait until I finish my command.\n",
            "The software I'm using also might pick up a lot of background noise, so if you think the text given to you is not for you, feel free to ignore it.\n",
            'For each phrase I give you, you will be able to call at most 10 functions.\n',
            f'The platform I am using is **{app.platform}**.\n'
        ],
        'role':
        'user',
    },
    {
        'parts': [
            'Okay, got it! I will try my best to interpret your intent with the informatino given to me and control your mouse and keyboard to execute the commands!'
        ],
        'role':
        'model',
    },
])
