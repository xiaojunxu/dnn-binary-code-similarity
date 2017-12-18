import cPickle as pkl
import json

ids = pkl.load(open('valid.ids'))
with open('valid.json', 'w') as outf:
    json.dump(ids, outf)

print (type(ids))
print (len(ids))
print (len(ids[0]))
print (len(ids[0][0]))
print (len(ids[0][0][0]))
print (len(ids[0][0][0][0]))
print (len(ids[0][0][0][0][0]))
