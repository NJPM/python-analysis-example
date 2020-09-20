


w = 3
x = "Hellow, world!"

if w > 2:
 print(x)
 print(type(x))
 print(x[6:8])
 print(x.upper())

w = 1

print(w > 2)
if w == 3:
 print(x)
else:
 print("Nope")

y = "1, 3, 6"

print(y)
print(y.split(","))
print("Y1 = " + y)

z = "1" in y
print(z)

def orders(name):
 order = name + " said \"{}\", but I meant {}."
 print(order.format(x, y))

orders("Bob")

mo = lambda a, b, c : a * b *c

print(mo(1, 2, 3))

farah = [2, 3, 4]

print(mo(farah[0], farah[1], farah[2]))

