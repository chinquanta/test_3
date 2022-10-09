from typing import List
from collections import Counter


class CountVectorizer():
    """
    Экземпляр класса CountVectorizer оцифровывает корпус входных текстов.

    Атрибут класса:
    __punct_signs - перечень ожидаемых знаков препинания в корпусе.

    Атрибуты экземпляра:
    input_corpus - входной корпус,
    tokenized_corpus - корпус с удаленными знаками препинания, разбитый на слова,
    dict - словарь корпуса (в виде множества),
    matrix - терм-документная матрица.
    """
    __punct_signs = '''!,.?();:'"'''


    @classmethod
    def clear_punct_sentence(cls, sentence: 'str') -> List[str]:
        """
        Возвращает исходное предложение, очищенное от знаков пунктуации,
        в виде списка слов.
        """
        for punct_sign in cls.__punct_signs:
            sentence = sentence.replace(punct_sign, ' ')
        return sentence.split()


    def __init__(self):
        self.input_corpus = None
        self.tokenized_corpus = None
        self.dict = None
        self.matrix = None


    def get_feature_names(self) -> List[str]:
        """
        Возвращает словарь уникальных слов из корпуса
        """
        return self.dict


    def clear_punct_corpus(self, corpus: List[str]) -> List[List[str]]:
        """
        Возвращает корпус с предложениями очищенными от знаков пунктуации.
        Каждое предложение - список слов в виде строк.
        """
        return list(map(self.clear_punct_sentence, corpus))


    def fit_transform(self, corpus: List[str] = []) -> List[List[int]]:
        """
        Создает атрибуты векторайзера по входному корпусу.

        Возвращает терм-документную матрицу.

        """
        self.input_corpus = corpus[:]
        self.tokenized_corpus = self.clear_punct_corpus(corpus)
        self.create_dict()
        self.create_matrix()
        return self.matrix


    def create_matrix(self):
        """
        Создает документарную матрицу для экземпляра класса
        """
        self.matrix = []
        if self.tokenized_corpus is not []:
            for item in self.tokenized_corpus:
                item_cnt = Counter(item)
                self.matrix.append([item_cnt[tocken] for tocken in self.dict])


    def create_dict(self):
        """
        Создает словарь токенов для экземпляра класса.
        Сортирует словарь лексикографически.
        """
        self.dict = set()
        self.dict.update(*self.tokenized_corpus)
        self.dict = list(self.dict)
        self.dict.sort()


if __name__ == '__main__':
    corpus = ['asd, fgt opu!', 'asdf', ' 6-7 ']
    cv = CountVectorizer()
    X = cv.fit_transform(corpus)
    print('Тестируем на нормальном корпусе:')
    print('Матрица ', X)
    print('Словарь ', cv.get_feature_names())
    print()
    print('Тестируем на пустом корпусе:')
    corpus = []
    X = cv.fit_transform(corpus)
    print('Матрица ', X)
    print('Словарь ', cv.get_feature_names())
    print()
    print('Тестируем неверный порядок вывода данных:')
    cv = CountVectorizer()
    corpus = ['asd, fgt opu!', 'asdf', ' 6-7 ']
    print('Словарь ', cv.get_feature_names())
    X = cv.fit_transform(corpus)
    print('Матрица ', X)
    print('Словарь ', cv.get_feature_names())