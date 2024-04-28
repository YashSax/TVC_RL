from utils import *

p = Persistence(3)
print(p.is_on())
p.on()
print(p.is_on())
p.update()
print(p.is_on())
p.update()
p.update()
print(p.is_on())
p.update()
print(p.is_on())
