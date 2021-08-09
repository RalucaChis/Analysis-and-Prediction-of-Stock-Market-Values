import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

def sa():
    dataset = pd.read_csv("/datasets/sa-complete-may_march.csv")
    dataset = dataset.sort_values(by=['Date'])

    plt.clf()
    plt.plot(dataset.loc[::, 'Date'], dataset[::]['Sentiment'], color='black')
    plt.title('Sentiment Analysis')
    plt.xlabel('Date')
    plt.ylabel('Sentiment value')
    # plt.legend()
    plt.savefig('D:\licenta\dash-project\plots\SA.png')

# sa()

def arrow_plot():
    dataset = pd.read_csv("D:\licenta\django-backend\\reddit_service\datasets\\twitter_data.csv")
    s1 = sum(dataset[len(dataset) - 2::]['SentimentValue']) / 2
    s2 = sum(dataset[len(dataset) - 4:len(dataset) - 2:]['SentimentValue']) / 2

    if s1 > s2:
        diff = round((s1 - s2)*100, 2)
        text = "+ " + str(diff) + "%"
        img = Image.open("D:\\licenta\\django-backend\\charts_service\\plots\\SA_arrow_green.png")
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("C:\Windows\Fonts\Arial.ttf", 35)
        font2 = ImageFont.truetype("C:\Windows\Fonts\Arial.ttf", 75)
        draw.text((350, 100), "Sentiment analysis\nfor the last 5 days:", (34, 177, 76), font=font)
        draw.text((350, 200), text, (34, 177, 76), font=font2)
    else:
        diff = round((s2 - s1)*100, 2)
        text = "- " + str(diff) + "%"
        img = Image.open("D:\\licenta\\django-backend\\charts_service\\plots\\SA_arrow_red.png")
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("C:\Windows\Fonts\Arial.ttf", 35)
        font2 = ImageFont.truetype("C:\Windows\Fonts\Arial.ttf", 75)
        draw.text((350, 100), "Sentiment analysis\nfor the last 5 days:", (237, 28, 36), font=font)
        draw.text((350, 200), text, (237, 28, 36), font=font2)
    img.save('D:\\licenta\\django-backend\\charts_service\\plots\\SA_arrow_out.png')

# arrow_plot()
