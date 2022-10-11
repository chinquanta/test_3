from typing import List
from collections import Counter


class CountVectorizer:
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
    punct_signs = '''!,.?();:'"'''


    @classmethod
    def clear_punct_sentence(cls, sentence: str) -> List[str]:
        """
        Возвращает исходное предложение, очищенное от знаков пунктуации,
        в виде списка слов.
        """
        for punct_sign in cls.punct_signs:
            sentence = sentence.replace(punct_sign, ' ')
        return sentence.split()


    def __init__(self):
        self.input_corpus = None
        self.tokenized_corpus = None
        self.dictionary = None
        self.matrix = None


    def get_feature_names(self) -> List[str]:
        """
        Возвращает словарь уникальных слов из корпуса
        """
        return self.dictionary


    def _clear_punct_corpus(self, corpus: List[str]) -> List[List[str]]:
        """
        Возвращает корпус с предложениями очищенными от знаков пунктуации.
        Каждое предложение - список слов в виде строк.
        """
        return list(map(self.clear_punct_sentence, corpus))


    def fit_transform(self, corpus: List[str]) -> List[List[int]]:
        """
        Создает атрибуты векторайзера по входному корпусу.

        Возвращает терм-документную матрицу.

        """
        if corpus:
            self.input_corpus = corpus
            self.tokenized_corpus = self._clear_punct_corpus(self.input_corpus)
            self._create_dict()
            self._create_matrix()
            return self.matrix
        else:
            raise ValueError("Корпус должен быть не пустым!")



    def _create_matrix(self):
        """
        Создает документарную матрицу для экземпляра класса
        """
        self.matrix = []
        if self.tokenized_corpus is not []:
            for item in self.tokenized_corpus:
                item_cnt = Counter(item)
                self.matrix.append([item_cnt[tocken] for tocken in self.dictionary])


    def _create_dict(self):
        """
        Создает словарь токенов для экземпляра класса.
        Сортирует словарь лексикографически.
        """
        self.dictionary = set()
        self.dictionary.update(*self.tokenized_corpus)
        self.dictionary = list(self.dictionary)
        self.dictionary.sort()


if __name__ == '__main__':
    print('Тестируем на нормальном корпусе с уникальными словами:')
    corpus = ['asd, fgt opu!', 'asdf', ' 6-7 ']
    etalon_matrix = [[0, 1, 0, 1, 1], [0, 0, 1, 0, 0], [1, 0, 0, 0, 0]]
    etalon_dictionary = ['6-7', 'asd', 'asdf', 'fgt', 'opu']
    cv = CountVectorizer()
    matrix = cv.fit_transform(corpus)
    dictionary = cv.get_feature_names()
    assert dictionary ==  etalon_dictionary
    assert matrix == etalon_matrix
    print('    Корпус: ', corpus)
    print('    Матрица: ', matrix)
    print('    Словарь: ', dictionary)
    print()

    print('Тестируем на нормальном корпусе с пустым предложением:')
    corpus = ['asd, fgt opu!', 'asdf', ' 6-7 ', ' ,.  ']
    etalon_matrix = [[0, 1, 0, 1, 1], [0, 0, 1, 0, 0], [1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]]
    etalon_dictionary = ['6-7', 'asd', 'asdf', 'fgt', 'opu']
    cv = CountVectorizer()
    matrix = cv.fit_transform(corpus)
    dictionary = cv.get_feature_names()
    assert dictionary ==  etalon_dictionary
    assert matrix == etalon_matrix
    print('    Корпус: ', corpus)
    print('    Матрица: ', matrix)
    print('    Словарь: ', dictionary)
    print()

    print('Тестируем на нормальном корпусе с повторяющимися словами:')
    corpus = ['asd, fgt opu! asdf', 'asdf', ' 6-7 fgt, fgt']
    etalon_matrix = [[0, 1, 1, 1, 1], [0, 0, 1, 0, 0], [1, 0, 0, 2, 0]]
    etalon_dictionary = ['6-7', 'asd', 'asdf', 'fgt', 'opu']
    cv = CountVectorizer()
    matrix = cv.fit_transform(corpus)
    dictionary = cv.get_feature_names()
    assert dictionary ==  etalon_dictionary
    assert matrix == etalon_matrix
    print('    Корпус: ', corpus)
    print('    Матрица: ', matrix)
    print('    Словарь: ', dictionary)
    print()

    print('Тестируем на пустом корпусе:')
    corpus = []
    value_exception = "Корпус должен быть не пустым!"
    cv = CountVectorizer()
    try:
        X = cv.fit_transform(corpus)
    except Exception as exc:
        assert value_exception == exc.__str__()
        print(exc)