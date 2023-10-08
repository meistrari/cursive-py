from cursive import Cursive
import numpy as np

cursive = Cursive()

x1 = cursive.embed("""Pizza""")
                   
x2 = cursive.embed("""Cat""")

def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

print(cosine_similarity(x1, x2))