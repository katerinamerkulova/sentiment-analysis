import pickle

def load_test():
    import codecs
    import re
    test = codecs.open('test.csv', 'r', 'utf_8_sig').read()
    test = re.sub(r'[^\w]', ' ', test).lower()    # чистим от знаков
    test = test.split('review')[1::2]   # разделяем на список
    return test

# Собираем обучающую выборку
# Не запускать без надобности, это надолго :)
def parse():
    import re
    import requests
    import bs4
    import pandas as pd
    headers = {'accept': '*/*',
               'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36'}
    links = open('links.txt').read().split('\n')

    # получаем ссылки на бренды (группы) телефонов
    session = requests.Session()
    req = session.get('https://slonrekomenduet.com/category/smartphones.html', headers=headers)
    parser = bs4.BeautifulSoup(req.text, 'lxml')
    raw_links = parser.findAll('li')[:80]
    links = []
    for link in raw_links:
        link = re.search('/category/.+\.html', str(link.a))
        links.append(link.group(0))

    # получаем ссылки на модели телефонов
    model_links = []
    for link in links:
        session = requests.Session()
        req = session.get('https://slonrekomenduet.com{}'.format(link), headers=headers)
        parser = bs4.BeautifulSoup(req.text, 'lxml')
        raw_links = parser.findAll('div', {'class': 'model'})[:-10]
        for model_link in raw_links:
            model_links.append(re.search('/model/.+\.html', str(model_link.a)).group(0))

    # парсим страницы с отзывами
    reviews = []
    ratings = []
    for i, link in enumerate(model_links):
        print(i, round(i/1567, 4), link)    # наблюдаем ход парсинга
        session = requests.Session()
        req = session.get('https://slonrekomenduet.com{}'.format(link), headers=headers)
        parser = bs4.BeautifulSoup(req.text, 'lxml')
        try:
            n_pages = int(re.findall('page/\d', str(parser))[-1][-1])
        except IndexError:
            n_pages = 1
        for page in range(1, n_pages+1):
            review = parser.findAll('div', {'class': 'comment_text'})
            for rev in review:
                text = rev.text.lower()    # приводим к нижнему регистру
                text = text.replace('\xa0', ' ')
                text = re.sub(r'[^\w]', ' ', text)    # избавляемся от знаков препинания
                reviews.append(text)
            rating = parser.findAll('div', {'class': 'br-widget'})[1:len(review)+1]
            for rate in rating:
                ratings.append(len(rate.findAll('a', {'class': 'br-active'})))

    df = pd.DataFrame({'review': reviews, 'rating': ratings})
    df.to_csv('reviews.csv', sep=';')
    return df

def training():
    import pandas as pd
    # train dataset
    df = pd.read_csv('reviews.csv', sep=';').dropna()
    df['label'] = ['pos' if x > 3 else 'neg' for x in df['rating']]

    # Векторизация по частотности и обучение логистической регрессии
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('russian'))

    ppl = Pipeline([
        ('vectorizer', CountVectorizer(stop_words=stop_words, ngram_range=(1, 3))),
        ('classifier', LogisticRegression(solver='liblinear')),
    ])
    return ppl.fit(df['review'], df['label'])

with open('model.pickle', 'rb') as fid:
    ppl = pickle.load(fid)
    
def predict(test):
    return ppl.predict(test)