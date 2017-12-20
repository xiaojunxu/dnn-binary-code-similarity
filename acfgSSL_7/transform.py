import json

SOFTWARE=('openssl-1.0.1f-', 'openssl-1.0.1u-')
OPTIMIZATION=('-O0', '-O1','-O2','-O3')
COMPILER=('armeb-linux', 'i586-linux', 'mips-linux')
for sft in ('1.0.1f', '1.0.1u'):
    for comp in ('armeb', 'i586', 'mips'):
        for opt in ('O0', 'O1', 'O2','O3'):
            print (sft, comp, opt)
            with open('openssl-%s-%s-linux-%sv54.txt'%(sft,comp,opt)) as inf, \
                    open('openssl-%s-%s-linux-%sv54.json'%(sft,comp,opt), 'w') as outf:
                g_num = int(inf.readline().strip())
                for _ in range(g_num):
                    graph = {}
                    line = inf.readline().strip()
                    n_num, lab = line.split(' ')
                    n_num = int(n_num)
                    fname, src = lab.split('||')
                    graph['n_num'] = n_num
                    graph['fname'] = fname
                    graph['src'] = src
                    graph['features'] = []
                    graph['succs'] = []
                    for n in range(n_num):
                        line = inf.readline().strip().split(' ')
                        graph['features'].append([float(x) for x in line[:7]])
                        graph['succs'].append([int(x) for x in line[8:]])
                    outf.write(json.dumps(graph)+'\n')
