"""
MonkeyType-inspired Typing Race Program
======================================

A terminal-based typing speed test program that provides:
- Real-time visual feedback (green for correct, red for incorrect)
- 1-minute timer that starts when typing begins
- WPM calculation and performance metrics
- Real-time plotting of typing speed
- ESC to quit, TAB+ENTER to restart

Author: DC
Dependencies: colorama, keyboard, matplotlib, threading
"""

import random
import time
import threading
import os
import sys
import subprocess
import argparse
from collections import deque
from typing import List, Optional
from colorama import Fore, Back, Style
import matplotlib.pyplot as plt
import keyboard

# Constants
TEST_DURATION = 60  # seconds (will be updated by command line args)
DEFAULT_WORD_COUNT = 60  # Base word count for 60 second test
CHARS_PER_WORD = 5
WPM_TRACKING_INTERVAL = 1
MAX_WPM_HISTORY = 60
PLOT_FILENAME = 'typing_speed_plot.png'
PB_FILENAME = 'personal_best.txt'
MOVING_AVERAGE_WINDOW = 5


def check_and_import_dependencies():
    """Check and import all 'nonstandard' required dependencies."""
    missing_libs = []
    
    try:
        import colorama
        colorama.init(autoreset=True)
    except ImportError:
        missing_libs.append('colorama')
    
    try:
        import keyboard
    except ImportError:
        missing_libs.append('keyboard')
    
    if missing_libs:
        print("Error: Missing required libraries. Install with:")
        for lib in missing_libs:
            print(f"  pip install {lib}")
        sys.exit(1)

check_and_import_dependencies()

# Word dictionary with categories for typing test
WORD_DICTIONARY = {
    "common": [
        # Most frequent English words (1-3 letters, very basic)
        "the", "be", "of", "and", "a", "to", "in", "he", "have", "it", "that", "for", 
        "they", "with", "as", "not", "on", "she", "at", "by", "this", "we", "you", 
        "do", "but", "from", "or", "which", "one", "would", "all", "will", "there", 
        "say", "who", "make", "when", "can", "more", "if", "no", "man", "out", "other",
        "so", "what", "time", "up", "go", "her", "my", "me", "him", "his", "had", 
        "has", "been", "an", "was", "are", "is", "am", "get", "got", "see", "saw",
        "come", "came", "give", "gave", "take", "took", "know", "knew", "think", "thought",
        "tell", "told", "feel", "felt", "find", "found", "keep", "kept", "let", "put",
        "run", "ran", "walk", "walked", "talk", "talked", "work", "worked", "play", "played",
        "help", "helped", "want", "wanted", "need", "needed", "try", "tried", "ask", "asked",
        "old", "new", "first", "last", "long", "great", "little", "own", "other", "right",
        "big", "high", "different", "small", "large", "next", "early", "young", "important",
        "few", "public", "bad", "same", "able", "call", "use", "her", "each", "which"
    ],
    "medium": [
        # Common words (4-7 letters, moderate frequency)
        "about", "than", "into", "could", "state", "only", "year", "some", 
        "take", "come", "these", "know", "like", "then", "first", 
        "any", "work", "now", "may", "such", "give", "over", "think", "most", "even", 
        "find", "day", "also", "after", "way", "many", "must", "look", "before",
        "being", "going", "doing", "having", "making", "getting", "coming", "looking",
        "working", "thinking", "feeling", "saying", "telling", "asking", "trying", "helping",
        "learning", "teaching", "reading", "writing", "listening", "speaking", "watching", "playing",
        "living", "moving", "walking", "running", "sitting", "standing", "sleeping", "eating",
        "drinking", "cooking", "cleaning", "building", "creating", "designing", "planning", "organizing",
        "managing", "leading", "following", "understanding", "explaining", "describing", "discussing", "arguing",
        "house", "water", "school", "family", "story", "money", "place", "music", "where",
        "person", "today", "always", "every", "world", "still", "between", "during", "example",
        "without", "system", "group", "often", "around", "problem", "rather", "enough", "quite"
    ],
    "advanced": [
        # More complex words (6-10 letters, lower frequency)
        "through", "should", "people", "because", "good", "each", "those", 
        "seem", "place", "little", "world", "very", "still", "back", "down", "where", "much", "well",
        "important", "different", "large", "small", "local", "certain", "available", "political",
        "economic", "social", "national", "international", "personal", "professional", "educational", "medical",
        "financial", "cultural", "historical", "scientific", "technological", "environmental", "psychological", "philosophical",
        "beautiful", "wonderful", "excellent", "amazing", "incredible", "fantastic", "outstanding", "remarkable",
        "significant", "substantial", "considerable", "enormous", "tremendous", "extraordinary", "magnificent", "spectacular",
        "challenging", "difficult", "complex", "complicated", "sophisticated", "advanced", "innovative", "creative",
        "effective", "efficient", "successful", "powerful", "influential", "necessary", "essential", "potential",
        "community", "development", "government", "education", "business", "service", "research", "information",
        "technology", "experience", "industry", "economic", "society", "organization", "university", "literature",
        "security", "decision", "relationship", "knowledge", "opportunity", "management", "particular", "political"
    ],
    "technical": [
        # Programming and technology terms
        "program", "system", "computer", "software", "hardware", "network", "database", 
        "algorithm", "function", "variable", "method", "class", "object", "string", 
        "integer", "boolean", "array", "loop", "condition", "exception", "interface",
        "framework", "library", "module", "package", "import", "export", "compile", "execute",
        "debug", "test", "deploy", "server", "client", "protocol", "encryption", "security",
        "authentication", "authorization", "validation", "configuration", "installation", "documentation", "repository", "version",
        "branch", "merge", "commit", "push", "pull", "clone", "fork", "issue",
        "feature", "bugfix", "patch", "release", "update", "upgrade", "migration", "backup",
        "restore", "monitor", "analyze", "optimize", "performance", "scalability", "reliability", "availability",
        "architecture", "design", "pattern", "principle", "methodology", "practice", "standard", "convention",
        "api", "url", "http", "https", "json", "xml", "html", "css", "javascript", "python",
        "developer", "programming", "application", "infrastructure", "middleware", "frontend", "backend", "database",
        "machine", "learning", "artificial", "intelligence", "cloud", "container", "kubernetes", "docker",
        "microservice", "bandwidth", "firewall", "router", "switch", "ethernet", "wireless", "bluetooth"
    ],
    "challenging": [
        # Very long and complex words (8+ letters, specialized vocabulary)
        "character", "experience", "relationship", "responsibility", "opportunity", "communication", "organization", "information",
        "administration", "recommendation", "investigation", "presentation", "representation", "interpretation", "implementation", "demonstration",
        "concentration", "consideration", "determination", "explanation", "preparation", "registration", "celebration", "observation",
        "imagination", "inspiration", "motivation", "dedication", "appreciation", "recognition", "understanding", "achievement",
        "development", "improvement", "establishment", "requirement", "arrangement", "management", "government", "environment",
        "knowledge", "intelligence", "excellence", "performance", "appearance", "difference", "reference", "preference",
        "conference", "influence", "violence", "evidence", "confidence", "independence", "dependence", "existence",
        "resistance", "assistance", "persistence", "consistency", "efficiency", "proficiency", "sufficiency", "deficiency",
        "democracy", "bureaucracy", "aristocracy", "meritocracy", "philosophy", "psychology", "technology", "biology",
        "extraordinary", "responsibility", "incomprehensible", "transformation", "revolutionary", "multiplication", "characterization",
        "internationalization", "constitutionality", "interdisciplinary", "entrepreneurship", "telecommunications", "manufacturing",
        "pharmaceutical", "sustainability", "metropolitan", "congressional", "contemporary", "retrospective", "comprehensive",
        "cardiovascular", "administrative", "sophisticated", "complementary", "unprecedented", "archaeological", "geographical"
    ]
}


class GameStats:
    """Handles all statistics and performance tracking."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.correct_chars = 0
        self.total_chars = 0
        self.wpm_data = deque(maxlen=MAX_WPM_HISTORY)
        self.time_stamps = deque(maxlen=MAX_WPM_HISTORY)
        self.accuracy_data = deque(maxlen=MAX_WPM_HISTORY)
        self.words_typed_correctly: List[bool] = []
        # Track accuracy for each character position (first attempt only)
        self.char_accuracy_map: dict = {}  # (word_index, char_index) -> bool
        # Simple accuracy tracking - starts at 100%, can only decrease
        self.current_accuracy = 100.0
    
    def add_wpm_data_point(self, wpm: float, elapsed_time: float):
        """Add a WPM data point for tracking."""
        self.wpm_data.append(wpm)
        self.time_stamps.append(elapsed_time)
        # Also track accuracy at this point
        accuracy = self.calculate_accuracy()
        self.accuracy_data.append(accuracy)
    
    def record_character_attempt(self, word_index: int, char_index: int, is_correct: bool):
        """Record the first attempt at typing a character position."""
        position_key = (word_index, char_index)
        if position_key not in self.char_accuracy_map:
            # Only record the first attempt
            self.char_accuracy_map[position_key] = is_correct
            self.total_chars += 1
            if is_correct:
                self.correct_chars += 1
            
            # Calculate current real accuracy and only update if it's lower
            if self.total_chars > 0:
                real_accuracy = (self.correct_chars / self.total_chars) * 100
                self.current_accuracy = min(self.current_accuracy, real_accuracy)
    
    def calculate_accuracy(self) -> float:
        """Calculate typing accuracy percentage - can only decrease."""
        return self.current_accuracy
    
    def get_words_completed(self) -> int:
        """Get number of words typed correctly."""
        return sum(1 for correct in self.words_typed_correctly if correct)


class DisplayManager:
    """Handles all display and UI-related functions."""
    
    @staticmethod
    def clear_screen():
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    @staticmethod
    def move_cursor_to_top():
        """Move cursor to top of screen without clearing."""
        # For Windows, we'll use a more reliable method
        if os.name == 'nt':
            # Windows - use cls for more reliable clearing
            os.system('cls')
        else:
            # Unix/Linux/Mac
            print('\033[H\033[2J', end='')
    
    @staticmethod
    def display_header(is_test_active: bool, start_time: Optional[float] = None):
        """Display the game header with timer information."""
        print(f"{Fore.CYAN}=== TYPING RACE TEST ==={Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Press ESC to quit | TAB to restart{Style.RESET_ALL}")
        
        if is_test_active and start_time:
            elapsed = time.time() - start_time
            remaining = max(0, TEST_DURATION - elapsed)
            print(f"{Fore.MAGENTA}Time remaining: {remaining:.1f}s{Style.RESET_ALL}")
        else:
            print(f"{Fore.GREEN}Start typing to begin the test!{Style.RESET_ALL}")
        print()  # Add spacing
    
    @staticmethod
    def display_words_with_progress(
        words: List[str], 
        current_word_index: int, 
        current_char_index: int, 
        words_typed_correctly: List[bool],
        current_word_correct: bool, 
        typed_chars: List[str]
    ):
        """Display words with color-coded progress."""
        print("=" * 60)
        
        display_text = ""
        words_per_line = 10  # Show 10 words per line for better formatting
        
        for i, word in enumerate(words):
            if i < current_word_index:
                # Word already completed
                if i < len(words_typed_correctly) and words_typed_correctly[i]:
                    display_text += f"{Fore.GREEN}{word}{Style.RESET_ALL} "
                else:
                    display_text += f"{Fore.RED}{word}{Style.RESET_ALL} "
            elif i == current_word_index:
                # Current word being typed
                if current_char_index == 0:
                    display_text += f"{Back.BLUE}{Fore.WHITE}{word}{Style.RESET_ALL} "
                else:
                    typed_part = ''.join(typed_chars)
                    
                    # Show what user has typed vs what should be typed
                    correct_part = ""
                    incorrect_part = ""
                    
                    # Check each typed character
                    for j, typed_char in enumerate(typed_part):
                        if j < len(word) and typed_char == word[j]:
                            correct_part += typed_char
                        else:
                            incorrect_part += typed_char
                    
                    # Remaining part of the word
                    remaining_part = word[len(typed_part):] if len(typed_part) < len(word) else ""
                    
                    # Build display string
                    if current_word_correct:
                        display_text += f"{Fore.GREEN}{correct_part}{Style.RESET_ALL}"
                    else:
                        display_text += f"{Fore.GREEN}{correct_part}{Style.RESET_ALL}{Back.RED}{Fore.WHITE}{incorrect_part}{Style.RESET_ALL}"
                    
                    if remaining_part:
                        display_text += f"{Back.BLUE}{Fore.WHITE}{remaining_part}{Style.RESET_ALL} "
                    else:
                        display_text += " "
            else:
                # Words not yet reached
                display_text += f"{Fore.WHITE}{word}{Style.RESET_ALL} "
            
            # Add line break every 10 words for better readability
            if (i + 1) % words_per_line == 0:
                display_text += "\n"
        
        print(display_text)
        print("=" * 60)
        print()  # Add spacing
    
    @staticmethod
    def display_current_stats(wpm: float, accuracy: float):
        """Display current typing statistics."""
        print(f"{Fore.CYAN}Current WPM: {wpm:.1f} | Accuracy: {accuracy:.1f}%{Style.RESET_ALL}")
        print()  # Add spacing
    
    @staticmethod
    def display_final_results(
        final_wpm: float, 
        accuracy: float, 
        elapsed_time: float, 
        words_completed: int
    ):
        """Display final test results."""
        print(f"{Fore.CYAN}=== TEST RESULTS ==={Style.RESET_ALL}")
        print(f"{Fore.GREEN}Final WPM: {final_wpm:.1f}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Accuracy: {accuracy:.1f}%{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}Test Duration: {elapsed_time:.1f} seconds{Style.RESET_ALL}")
        print(f"{Fore.BLUE}Words Completed: {words_completed}{Style.RESET_ALL}")
        print(f"\n{Fore.YELLOW}Press TAB to restart or ESC to quit{Style.RESET_ALL}")


class PlotManager:
    """Handles matplotlib plotting operations."""
    
    @staticmethod
    def create_wpm_plot(wpm_data: deque, time_stamps: deque, accuracy_data: deque) -> bool:
        """Create and display WPM progression plot with accuracy."""
        if len(wpm_data) < 2:
            return False
        
        try:
            # Use non-interactive backend to avoid threading issues
            import matplotlib
            matplotlib.use('Agg')  # Use non-GUI backend
            
            # Create subplot layout: WPM on top, accuracy on bottom
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            
            times = list(time_stamps)
            wpm_values = list(wpm_data)
            accuracy_values = list(accuracy_data)
            
            # Top plot: WPM
            ax1.plot(times, wpm_values, 'b-', linewidth=1, alpha=0.7, label='Raw WPM')
            
            # Calculate and plot moving average for WPM
            if len(wpm_values) >= MOVING_AVERAGE_WINDOW:
                moving_avg = PlotManager._calculate_moving_average(wpm_values, MOVING_AVERAGE_WINDOW)
                ax1.plot(times, moving_avg, 'r-', linewidth=2, label='Average WPM')
            
            ax1.set_ylabel('Words Per Minute (WPM)')
            ax1.set_title('Typing Speed Performance')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            ax1.set_ylim(0, 130)
            
            # Bottom plot: Accuracy
            ax2.plot(times, accuracy_values, 'g-', linewidth=2, label='Accuracy')
            
            # Calculate and plot moving average for accuracy
            if len(accuracy_values) >= MOVING_AVERAGE_WINDOW:
                accuracy_moving_avg = PlotManager._calculate_moving_average(accuracy_values, MOVING_AVERAGE_WINDOW)
                ax2.plot(times, accuracy_moving_avg, 'orange', linewidth=2, label='Average Accuracy')
            
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('Accuracy (%)')
            ax2.set_title('Typing Accuracy')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            ax2.set_ylim(0, 100)
            
            plt.tight_layout()
            
            # Save plot instead of showing it to avoid threading issues
            plt.savefig(PLOT_FILENAME, dpi=300, bbox_inches='tight')
            plt.close()  # Close the figure to free memory
            
            print(f"{Fore.GREEN}WPM plot saved as '{PLOT_FILENAME}'{Style.RESET_ALL}")
            
            # Open the saved plot image using the system's default image viewer
            try:
                if os.name == 'nt':  # Windows
                    os.startfile(PLOT_FILENAME)
                elif os.name == 'posix':  # macOS and Linux
                    if sys.platform == 'darwin':  # macOS
                        subprocess.run(['open', PLOT_FILENAME])
                    else:  # Linux
                        subprocess.run(['xdg-open', PLOT_FILENAME])
                print(f"{Fore.CYAN}Opening your typing speed chart...{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.YELLOW}Chart saved as '{PLOT_FILENAME}' - please open manually{Style.RESET_ALL}")
            
            return True
            
        except Exception as e:
            print(f"{Fore.RED}Could not create plot: {e}{Style.RESET_ALL}")
            return False
    
    @staticmethod
    def _calculate_moving_average(values: List[float], window_size: int) -> List[float]:
        """Calculate moving average for the given values."""
        moving_avg = []
        for i in range(len(values)):
            start_idx = max(0, i - window_size + 1)
            end_idx = i + 1
            avg = sum(values[start_idx:end_idx]) / (end_idx - start_idx)
            moving_avg.append(avg)
        return moving_avg
    
    @staticmethod
    def close_all_plots():
        """Close all matplotlib windows."""
        try:
            plt.close('all')
        except Exception:
            pass  # Ignore errors when closing plots


class TypingRaceGame:
    """Main typing race game class."""
    
    def __init__(self):
        self.words: List[str] = []
        self.current_word_index = 0
        self.current_char_index = 0
        self.typed_chars: List[str] = []  # Track actual typed characters for current word
        self.start_time: Optional[float] = None
        self.timer_thread: Optional[threading.Thread] = None
        self.is_test_active = False
        self.test_ended = False
        self.current_word_correct = True
        
        # Initialize managers
        self.stats = GameStats()
        self.display = DisplayManager()
        self.plot_manager = PlotManager()
        
        # Thread safety
        self._stats_lock = threading.Lock()
        
        # Load personal best
        self.personal_best = self.load_personal_best()
        
    def load_personal_best(self) -> float:
        """Load personal best WPM from file."""
        try:
            if os.path.exists(PB_FILENAME):
                with open(PB_FILENAME, 'r') as f:
                    return float(f.read().strip())
        except Exception:
            pass
        return 0.0
    
    def save_personal_best(self, wpm: float):
        """Save new personal best WPM to file."""
        try:
            with open(PB_FILENAME, 'w') as f:
                f.write(f"{wpm:.1f}")
        except Exception as e:
            print(f"{Fore.RED}Could not save personal best: {e}{Style.RESET_ALL}")
    
    def check_and_update_personal_best(self, final_wpm: float) -> tuple[bool, float]:
        """Check if current score is a new personal best and update if so."""
        old_best = self.personal_best
        if final_wpm > self.personal_best:
            self.personal_best = final_wpm
            self.save_personal_best(final_wpm)
            return True, old_best
        return False, old_best
    
    def generate_words(self, count: int = None, difficulty: str = "mixed"):
        """Generate random words for the typing test."""
        # Calculate word count based on test duration if not specified
        if count is None:
            # Scale words based on test duration (1 word per second as baseline)
            count = max(30, int(TEST_DURATION * 1.0))  # Minimum 30 words, scale with duration
        
        if difficulty == "mixed":
            # Mix words from different categories
            all_words = []
            for category_words in WORD_DICTIONARY.values():
                all_words.extend(category_words)
            self.words = random.sample(all_words, min(count, len(all_words)))
        else:
            # Use words from specific category
            if difficulty in WORD_DICTIONARY:
                category_words = WORD_DICTIONARY[difficulty]
                self.words = random.sample(category_words, min(count, len(category_words)))
            else:
                # Fall back to mixed if invalid difficulty
                all_words = []
                for category_words in WORD_DICTIONARY.values():
                    all_words.extend(category_words)
                self.words = random.sample(all_words, min(count, len(all_words)))
        
        self.current_word_index = 0
        self.current_char_index = 0
        self.typed_chars = []
        self.current_word_correct = True
        self.stats.reset()
        
    def calculate_current_wpm(self) -> float:
        """Calculate current WPM based on correctly typed characters."""
        if not self.start_time:
            return 0.0
        
        elapsed_minutes = (time.time() - self.start_time) / 60
        if elapsed_minutes == 0:
            return 0.0
        
        # Count characters in correctly typed words + current correct chars
        with self._stats_lock:
            correct_word_chars = sum(
                len(word) + 1 for i, word in enumerate(self.words) 
                if i < len(self.stats.words_typed_correctly) and self.stats.words_typed_correctly[i]
            )
            
            if self.current_word_correct and self.current_word_index < len(self.words):
                correct_word_chars += self.current_char_index
        
        words_typed = correct_word_chars / CHARS_PER_WORD
        return words_typed / elapsed_minutes if elapsed_minutes > 0 else 0.0
    
    def display_game_state(self):
        """Display the current game state."""
        # Use full screen clearing for reliable display on all platforms
        self.display.clear_screen()
        self.display.display_header(self.is_test_active, self.start_time)
        self.display.display_words_with_progress(
            self.words, self.current_word_index, self.current_char_index,
            self.stats.words_typed_correctly, self.current_word_correct, self.typed_chars
        )
        
        if self.is_test_active:
            current_wpm = self.calculate_current_wpm()
            accuracy = self.stats.calculate_accuracy()
            self.display.display_current_stats(current_wpm, accuracy)
    
    def start_timer(self):
        """Start the test timer and tracking threads."""
        if self.is_test_active:
            return
        
        self.start_time = time.time()
        self.is_test_active = True
        self.test_ended = False
        
        # Start WPM tracking thread
        def track_wpm():
            while self.is_test_active and not self.test_ended:
                time.sleep(WPM_TRACKING_INTERVAL)
                if self.is_test_active and self.start_time:
                    current_wpm = self.calculate_current_wpm()
                    elapsed = time.time() - self.start_time
                    with self._stats_lock:
                        self.stats.add_wpm_data_point(current_wpm, elapsed)
        
        threading.Thread(target=track_wpm, daemon=True).start()
        
        # Start countdown timer
        def countdown():
            time.sleep(TEST_DURATION)
            if self.is_test_active:
                self.end_test()
        
        self.timer_thread = threading.Thread(target=countdown, daemon=True)
        self.timer_thread.start()
    
    def end_test(self):
        """End the typing test and show results."""
        self.is_test_active = False
        self.test_ended = True
        self.show_results()
    
    def show_results(self):
        """Display final results and WPM plot."""
        self.display.clear_screen()
        
        final_wpm = self.calculate_current_wpm()
        accuracy = self.stats.calculate_accuracy()
        elapsed_time = time.time() - self.start_time if self.start_time else 0.0
        words_completed = self.stats.get_words_completed()
        
        # Check for personal best
        is_new_pb, old_best = self.check_and_update_personal_best(final_wpm)
        
        self.display.display_final_results(final_wpm, accuracy, elapsed_time, words_completed)
        
        # Display personal best information
        if is_new_pb:
            print(f"{Fore.YELLOW}🎉 NEW PERSONAL BEST! 🎉{Style.RESET_ALL}")
            if old_best > 0:
                print(f"{Fore.GREEN}Previous best: {old_best:.1f} WPM{Style.RESET_ALL}")
        else:
            print(f"{Fore.CYAN}Personal Best: {self.personal_best:.1f} WPM{Style.RESET_ALL}")
            if self.personal_best > final_wpm:
                difference = self.personal_best - final_wpm
                print(f"{Fore.YELLOW}You're {difference:.1f} WPM away from your best!{Style.RESET_ALL}")
        print()
        
        # Create WPM plot and save it
        with self._stats_lock:
            plot_created = self.plot_manager.create_wpm_plot(self.stats.wpm_data, self.stats.time_stamps, self.stats.accuracy_data)
            if plot_created:
                print(f"{Fore.CYAN}Your typing speed chart has been saved to view your progress!{Style.RESET_ALL}")
            print()
    
    def handle_keypress(self, key: str) -> bool:
        """Handle individual keypress events."""
        try:
            if key == 'esc':
                self.is_test_active = False
                self.test_ended = True
                self.plot_manager.close_all_plots()
                return False  # Exit the program
            
            if key == 'tab':
                self.restart_test()
                return True
            
            # Don't process keys if test is ended (except ESC and TAB which are handled above)
            if self.test_ended:
                return True
            
            # Start timer on first keypress
            if not self.is_test_active:
                self.start_timer()
            
            if self.current_word_index >= len(self.words):
                return True
            
            current_word = self.words[self.current_word_index]
            
            if key == 'space':
                self._handle_space_key()
            elif key == 'backspace':
                self._handle_backspace_key()
            elif len(key) == 1 and key.isprintable():
                self._handle_character_key(key, current_word)
            
            self.display_game_state()
            return True
            
        except Exception as e:
            print(f"Error handling keypress: {e}")
            return True
    
    def _handle_space_key(self):
        """Handle space key press (move to next word)."""
        current_word = self.words[self.current_word_index]
        # Check if typed word matches exactly
        typed_word = ''.join(self.typed_chars)
        word_correct = (typed_word == current_word)
        
        with self._stats_lock:
            self.stats.words_typed_correctly.append(word_correct)
        
        self.current_word_index += 1
        self.current_char_index = 0
        self.typed_chars = []
        self.current_word_correct = True
    
    def _handle_backspace_key(self):
        """Handle backspace key press."""
        if self.current_char_index > 0:
            self.current_char_index -= 1
            if self.typed_chars:
                self.typed_chars.pop()
            self._check_current_word_correctness()
    
    def _handle_character_key(self, key: str, current_word: str):
        """Handle regular character key press."""
        # Add the typed character to our tracking
        self.typed_chars.append(key)
        
        # Check if this character is correct and record first attempt for accuracy
        is_correct = (self.current_char_index < len(current_word) and 
                     key == current_word[self.current_char_index])
        
        with self._stats_lock:
            self.stats.record_character_attempt(
                self.current_word_index, 
                self.current_char_index, 
                is_correct
            )
        
        self.current_char_index += 1
        self._check_current_word_correctness()
    
    def _check_current_word_correctness(self):
        """Check if the current word typed so far is correct."""
        if self.current_word_index >= len(self.words):
            return
        
        current_word = self.words[self.current_word_index]
        typed_so_far = ''.join(self.typed_chars)
        
        # Check if what's typed so far matches the expected characters
        self.current_word_correct = True
        
        # If typed more characters than the word has, it's wrong
        if len(typed_so_far) > len(current_word):
            self.current_word_correct = False
        else:
            # Check each character that has been typed
            for i in range(len(typed_so_far)):
                if i >= len(current_word) or typed_so_far[i] != current_word[i]:
                    self.current_word_correct = False
                    break
    
    def restart_test(self):
        """Restart the typing test with new words."""
        self.is_test_active = False
        self.test_ended = False
        
        # Close any existing matplotlib windows
        self.plot_manager.close_all_plots()
        
        self.generate_words()
        self.display_game_state()
    
    def run(self):
        """Main game loop."""
        self.display.clear_screen()
        print(f"{Fore.CYAN}Welcome to Python Typing Race!{Style.RESET_ALL}")
        
        # Calculate and display word count
        word_count = max(30, int(TEST_DURATION * 1.0))
        print(f"{Fore.YELLOW}Test Duration: {TEST_DURATION}s | Word Count: {word_count} words{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Generating words for your typing test...{Style.RESET_ALL}")
        print()
        
        self.generate_words()
        self.display_game_state()
        
        # Main input loop
        try:
            while True:
                event = keyboard.read_event()
                
                if event.event_type == keyboard.KEY_DOWN:
                    key_name = event.name
                    
                    # Skip modifier keys
                    if key_name in ['shift', 'ctrl', 'alt', 'cmd', 'enter']:
                        continue
                    
                    # Process the keypress
                    should_continue = self.handle_keypress(key_name)
                    if not should_continue:
                        break
                        
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Game interrupted by user{Style.RESET_ALL}")
        except Exception as e:
            print(f"\n{Fore.RED}Error: {e}{Style.RESET_ALL}")
        finally:
            # Ensure clean exit
            self.is_test_active = False
            self.test_ended = True
            self.plot_manager.close_all_plots()
            self.display.clear_screen()
            print(f"{Fore.CYAN}Thanks for playing Python Typing Race!{Style.RESET_ALL}")
            print(f"{Fore.GREEN}Game exited successfully.{Style.RESET_ALL}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MonkeyType-inspired terminal typing speed test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python typegame.py                    # Default: 60 second test
  python typegame.py 30                 # 30 second test
  python typegame.py 120 myrecord.txt   # 2 minute test with custom PB file
        """
    )
    
    parser.add_argument(
        'duration', 
        type=int, 
        nargs='?', 
        default=60,
        help='Test duration in seconds (default: 60)'
    )
    
    parser.add_argument(
        'pbfile', 
        type=str, 
        nargs='?', 
        default='personal_best.txt',
        help='Personal best record file path (default: personal_best.txt)'
    )
    
    return parser.parse_args()


def main():
    """Main function to run the typing race game."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Update global constants based on arguments
        global TEST_DURATION, PB_FILENAME
        TEST_DURATION = args.duration
        PB_FILENAME = args.pbfile
        
        print(f"{Fore.CYAN}Typing Race Configuration:{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Test Duration: {TEST_DURATION} seconds{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Personal Best File: {PB_FILENAME}{Style.RESET_ALL}")
        print()
        
        game = TypingRaceGame()
        game.run()
    except Exception as e:
        print(f"Failed to start game: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

