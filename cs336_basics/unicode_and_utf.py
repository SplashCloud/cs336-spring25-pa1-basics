# Unicode
# you can use `ord()` to convert a unicode character to a code point
# and use `chr()` to convert it back to the character
unicode = ord('牛')
print(unicode)
print(chr(unicode))

# UFT-8
# encode the code point of Unicode into a byte sequence
test_string = "hello! こんにちは!"
print(test_string.encode(encoding='utf-8'))
print(test_string.encode(encoding='utf-16'))
print(test_string.encode(encoding='utf-32'))


def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    # not every byte in the sequence can correspond to a single character
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

def decode_utf8_bytes_to_str_right(bytestring: bytes):
    return bytestring.decode("utf-8")

utf8_bytes = test_string.encode()
print(decode_utf8_bytes_to_str_right(utf8_bytes))
print(decode_utf8_bytes_to_str_wrong(utf8_bytes))