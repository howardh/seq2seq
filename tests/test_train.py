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

if __name__ == '__main__':
    unittest.main()
