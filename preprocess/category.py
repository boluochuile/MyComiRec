import sys

name = 'ml-10m'
if len(sys.argv) > 1:
    name = sys.argv[1]

item_cate = {}
item_map = {}
cate_map = {}
with open('../data/%s_data/%s_item_map.txt' % (name, name), 'r') as f:
    for line in f:
        conts = line.strip().split(',')
        item_map[conts[0]] = conts[1]

cates = []
if name == 'ml-10m':
    with open('../data/ml-10m/movies.dat', 'r', encoding='ISO-8859-1') as f:
        for line in f:
            conts = line.strip().split('::')
            iid = conts[0]

            # if iid in item_map:
            #     for index, cate_name in enumerate(conts[2].split('|')):
            #         if cate_name not in cate_map:
            #             cate_map[cate_name] = len(cate_map) + 1
            #         if index == 0:
            #             item_cate[item_map[iid]] = cate_map[cate_name]
            #         else:
            #             item_cate[item_map[iid]] = str(item_cate[item_map[iid]]) + '|' + str(cate_map[cate_name])
            if iid not in item_map:
                continue
            # 取最后一个类型
            cate = conts[2].split('|')[-1]
            if cate not in cate_map:
                cate_map[cate] = len(cate_map) + 1
            item_cate[item_map[iid]] = cate_map[cate]

with open('../data/%s_data/%s_cate_map.txt' % (name, name), 'w') as f:
    for key, value in cate_map.items():
        f.write('%s,%s\n' % (key, value))
with open('../data/%s_data/%s_item_cate.txt' % (name, name), 'w') as f:
    for key, value in item_cate.items():
        f.write('%s,%s\n' % (key, value))
