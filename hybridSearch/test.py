import re

line = "[qUery 456]: SELECT * FROM users"
match = re.match(r'\[Query (\d+)\]:\s*(.*)$', line, re.IGNORECASE)
if match:
    print(match.group(1))  # 输出: 456
    print(match.group(2))  # 输出: SELECT * FROM users