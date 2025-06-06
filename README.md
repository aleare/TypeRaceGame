# Typing Race

A terminal-based typing speed test inspired by MonkeyType. Test your typing skills with real-time feedback and see your WPM improve over time!

## What's This About?

I got tired of switching to web browsers just to practice typing, so I built this little terminal app. It's simple but does everything I need - tracks my speed, shows mistakes in real-time, and gives me a nice graph at the end.

## Features

- **Real-time feedback** - correct letters in green, mistakes in red
- **Live WPM tracking** and accuracy percentage  
- **Performance graphs** showing speed and accuracy over time

## Quick Start

```bash
# Clone Repo
git clone https://github.com/aleare/TypeRaceGame.git

# Enter Folder
cd .\TypeRaceGame\

# Install requirements
pip install colorama keyboard matplotlib

# Run the game (default: 60 second test)
python typegame.py
```

That's it! Start typing to begin the test.

## Command Line Options

```bash
python typegame.py [duration] [pbfile]
```

**Arguments:**
- `duration` - Test duration in seconds (default: 60)
- `pbfile` - Personal best record file path (default: personal_best.txt)

**Examples:**
```bash
python typegame.py                      # 60 second test, default PB file
python typegame.py 30                   # 30 second test
python typegame.py 90 work_pb.txt       # 90 second test, custom PB file
python typegame.py 120 ~/typing/pb.txt  # 2 minute test, custom path
```

## Controls

- **Type normally** - the test starts automatically once you start typing
- **ESC** - quit the game
- **TAB** - restart with new words

## Requirements

- Python 3.6+
- colorama (for colored text)
- keyboard (for input handling)
- matplotlib (for the speed graph)

## Word Categories

The game pulls from different difficulty levels:
- **Common**: everyday words (the, and, have, etc.)
- **Medium**: action words and longer terms
- **Advanced**: descriptive and complex words
- **Technical**: programming and tech terms
- **Challenging**: longer words with complex spelling

## Why Not Just Use a online typeracer?

Good question! Sometimes you want to practice without opening a browser, or you're working in a terminal-heavy workflow. Or, as I did yesterday, you get zero internet connection from your ISP (yey!).

## Known Issues

- Might flicker slightly on some terminals (working on it)
- No support for custom word lists yet (maybe in v2?)
- Todo: Better difficulty selection \ games 

## Contributing

Found a bug? Want to add features? Feel free to open an issue or send a PR. The code is pretty straightforward - most of the logic is in the `TypingRaceGame` class.

## License

MIT - do whatever you want with it.

## Note

Code was refactored (functions names, comments, etc) using AI. Take care, sometimes comments could be wrong...

---

*Built with Python because it's quick.* 
