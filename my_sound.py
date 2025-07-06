import pygame

def play_sound(file_path):
    # Initialize the mixer
    pygame.mixer.init()
    # Load the audio file
    pygame.mixer.music.load(file_path)
    # Play the audio
    pygame.mixer.music.play()
    # Wait for the sound to finish playing
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)  # prevents high CPU usage

# Example usage:
if __name__ == "__main__":
    # Absolute path to your siren.mp3 file
    play_sound(r"C:\Users\saniy\Downloads\SilentAid\siren.mp3")

