import sys

lines = []
mems = []
fcs = []
with open(sys.argv[1], 'r') as f:
    line = f.readline()
    while line:
        lines.append(line)
        left, right = line.split(':')
        mem = 1.
        for n in right.split('*'):
            mem *= int(n)
        if left.startswith('FC'):
            if len(fcs) == 0:
                fcs.append(mem)
                mem *= (mems[-1]/4)
            else:
                fcs.append(mem)
                mem *= fcs[-2]
        mems.append(mem*4)
        line = f.readline()

with open(sys.argv[2], 'w') as f:
    total = 0.
    for l, m in zip(lines, mems):
        mem_in_mb = m / (1024.*1024.)
        total += mem_in_mb
        if mem_in_mb >= 1.:
            line = (l[:-1] + ' mem: %.4fMB\n') % (mem_in_mb)
        else:
            mem_in_kb = m / 1024.
            line = (l[:-1] + ' mem: %.4fKB\n') % (mem_in_kb)
        f.write(line)
    tot_usage = 'TOTAL USAGE~= %.4fMB\n' % total
    f.write(tot_usage)

