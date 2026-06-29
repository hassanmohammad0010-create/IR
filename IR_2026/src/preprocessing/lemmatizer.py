class Lemmatizer:

    @staticmethod
    def lemmatize(tokens: list) -> list[str]:

        result = []

        for token in tokens:
            lemma = token.lemma_.lower().strip()

            if not lemma:
                lemma = token.text.lower()

            if lemma == "-pron-":
                lemma = token.text.lower()

            result.append(lemma)

        return result
