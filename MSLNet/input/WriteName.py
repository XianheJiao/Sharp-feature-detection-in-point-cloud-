import re,os
def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if re.match(r'.*txt', f,re.I):
                fullname = os.path.join(root, f)
                yield fullname
testPath=r'test_all.txt'
if os.path.isfile(testPath):
    os.remove(testPath)
currentBase=os.getcwd()
names=[]
for txtPath in findAllFile(currentBase):
    name = re.split(r'\\', txtPath)[-1]
    name = re.split(r'\.txt', name)[0]
    names.append(name)
with open(testPath,'w') as w:
    for name in names:
        w.write(f'{name}\n')
