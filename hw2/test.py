from lib.util import * 
# data processing
with open(sys.argv[1], 'r') as f:
    model = json.load(f)
(W,B,Ave,SD) = (array(model['W']), model['B'], array(model['Ave']), array(model['SD'])) 
Xtest = testDataProcessHW2(Ave, SD, sys.argv[2])
Y_predict = []
for i in range(len(Xtest)):
    fxn = Sigmoid(inner(W,Xtest[i]) + B)
    if fxn >= 0.5:
        Y_predict.append(1)
    else:
        Y_predict.append(0)

with open( sys.argv[3], 'w') as csvfile:
    fieldnames = ['id','label']
    w = csv.DictWriter(csvfile,fieldnames=fieldnames)
    w.writeheader()
    for i in range(len(Y_predict)):
        w.writerow({'id':str(i + 1), 'label': Y_predict[i]})
