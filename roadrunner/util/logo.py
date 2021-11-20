class color:
    BOLD   = '\033[1m\033[48m'
    END    = '\033[0m'
    ORANGE = '\033[38;5;202m'
    BLACK  = '\033[38;5;240m'

def print_logo(subtitle=""):
    print(color.BOLD, end="")
    print(color.ORANGE, end="")
    print("                                                     ")
    print(" _____                   _        _   _        _     ")
    print("/  __ \                 (_)      | \ | |      | |    ")
    print("| /  \/  __ _  ___  ___  _   ___ |  \| |  ___ | |_   ")
    print("| |     / _` |/ __|/ __|| | / _ \| . ` | / _ \| __|  ")
    print("| \__/\| (_| |\__ \\\\__ \| ||  __/| |\  ||  __/| |_   ")
    print(" \____/ \__,_||___/|___/|_| \___|\_| \_/ \___| \__|  ")
    print("                                                     ")
    print("                                                     ")
    print(color.END)
    print(subtitle + "\n\n")
