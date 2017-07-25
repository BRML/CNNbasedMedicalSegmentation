import json

with open('brats_test_names.txt', 'r') as f:
    official_names = f.read()
    
official_names = official_names.split('\n')
official_ids = []

for o_name in official_names:
    o_id = o_name.split('.')[-2]
    official_ids.append(o_id)
    
with open('brats_test_info.json', 'r') as f:
    local_info = json.load(f)
    local_names = local_info['names']
    
local_ids = []
for l_name in local_names:
    l_id = l_name.split('.')[-2]
    local_ids.append(l_id)
  
print 'Comparing lists...\n'
  
count_l = 0
for l_id in local_ids:
    if l_id not in official_ids:
        print 'No matching official for: ', l_id
        count_l += 1
        
print 'Number of problems: %i/%i' % (count_l, len(local_ids))
print '-'*40
    
count_o = 0    
for o_id in official_ids:
    if o_id not in local_ids:
        print 'No matching local for: ', o_id
        count_o += 1

print 'Number of problems: %i/%i' % (count_o, len(official_ids))

print '\nCollecting matches...\n'
count_l = 0
for l_id in local_ids:
    if l_id in official_ids:
        print 'There is a matching official for: ', l_id
        count_l += 1
        
print 'Number of matches: %i/%i' % (count_l, len(local_ids))
