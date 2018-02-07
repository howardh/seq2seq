import nose
import train

def test_process_sentence():
    pairs = [
            (" a ", "a"),
            ("A.", "a ."),
            ("A b c.", "a b c ."),
            ("A b c!", "a b c !"),
            ("A b c?", "a b c ?"),
            ("Abc asdf cd?", "abc asdf cd ?"),
            ("it's", "it s"),
            ("I'm", "i m"),
    ]
    for sentence,expected in pairs:
        output = train.process_sentence("eng", sentence)
        yield nose.tools.eq_,expected,output,\
            'Input: "%s", Expected: "%s", Actual output: "%s"' % (sentence, expected, output)

def test_sentence_to_indices():
    sentences = ["", "a", "hello world !"]
    lang = train.Language("eng")
    for sentence in sentences:
        lang.add_sentence(sentence)
    for sentence in sentences:
        indices = train.sentence_to_indices(lang, sentence)
        sentence2 = train.indices_to_sentence(lang, indices)
        yield nose.tools.eq_, sentence, sentence2

if __name__ == '__main__':
    unittest.main()
