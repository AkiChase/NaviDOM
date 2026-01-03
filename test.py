import json


def main():
    with open("D:/globus/mind2web/a52fcf7a-50aa-4256-8796-654b3dc3adac/screenshot.json", "r") as f:
        data = json.load(f)
    t = data[6]
    pass


if __name__ == '__main__':
    main()
