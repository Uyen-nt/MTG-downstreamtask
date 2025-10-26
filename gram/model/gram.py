import sys, copy, pickle
import sys, pickle, os

if __name__ == '__main__':
    infile = sys.argv[1]       # synthetic seqs file
    seqFile = sys.argv[2]      # labels
    typeFile = sys.argv[3]     # tree prefix (types or pk)
    outFile = sys.argv[4]

    # Thử mở file text; nếu lỗi -> mở dạng binary
    try:
        infd = open(infile, 'r', encoding='utf-8')
        _ = infd.readline()
        infd.seek(0)
        is_binary = False
    except UnicodeDecodeError:
        infd = open(infile, 'rb')
        is_binary = True

    if is_binary:
        print(f"⚙️ File {infile} là pickle, mở bằng 'rb'")
    else:
        print(f"⚙️ File {infile} là text, mở bằng 'utf-8'")


    # Đọc dữ liệu
    infd = open(infile, 'r', encoding='utf-8')
    _ = infd.readline()

    seqs = pickle.load(open(seqFile, 'rb'))
    types = pickle.load(open(typeFile, 'rb'))

    startSet = set(types.keys())
    hitList, missList = [], []
    cat1count = cat2count = cat3count = cat4count = 0

    for line in infd:
        tokens = line.strip().split(',')
        if len(tokens) < 9:
            continue  # bỏ qua dòng không đủ trường

        icd9 = tokens[0][1:-1].strip()
        cat1, desc1 = tokens[1][1:-1].strip(), 'A_' + tokens[2][1:-1].strip()
        cat2, desc2 = tokens[3][1:-1].strip(), 'A_' + tokens[4][1:-1].strip()
        cat3, desc3 = tokens[5][1:-1].strip(), 'A_' + tokens[6][1:-1].strip()
        cat4, desc4 = tokens[7][1:-1].strip(), 'A_' + tokens[8][1:-1].strip()

        if icd9.startswith('E'):
            if len(icd9) > 4:
                icd9 = icd9[:4] + '.' + icd9[4:]
        else:
            if len(icd9) > 3:
                icd9 = icd9[:3] + '.' + icd9[3:]
        icd9 = 'D_' + icd9

        if icd9 not in types:
            missList.append(icd9)
        else:
            hitList.append(icd9)

        if desc1 not in types:
            cat1count += 1
            types[desc1] = len(types)
        if len(cat2) > 0 and desc2 not in types:
            cat2count += 1
            types[desc2] = len(types)
        if len(cat3) > 0 and desc3 not in types:
            cat3count += 1
            types[desc3] = len(types)
        if len(cat4) > 0 and desc4 not in types:
            cat4count += 1
            types[desc4] = len(types)
    infd.close()

    rootCode = len(types)
    types['A_ROOT'] = rootCode
    print(rootCode)
    print(f'cat1count: {cat1count}')
    print(f'cat2count: {cat2count}')
    print(f'cat3count: {cat3count}')
    print(f'cat4count: {cat4count}')
    print(f'Number of total ancestors: {cat1count + cat2count + cat3count + cat4count + 1}')
    print(f'miss count: {len(startSet - set(hitList))}')
    missSet = startSet - set(hitList)

    fiveMap, fourMap, threeMap, twoMap = {}, {}, {}, {}
    oneMap = {types[icd]: [types[icd], rootCode] for icd in missSet}

    infd = open(infile, 'r', encoding='utf-8')
    infd.readline()

    for line in infd:
        tokens = line.strip().split(',')
        if len(tokens) < 9:
            continue

        icd9 = tokens[0][1:-1].strip()
        cat1, desc1 = tokens[1][1:-1].strip(), 'A_' + tokens[2][1:-1].strip()
        cat2, desc2 = tokens[3][1:-1].strip(), 'A_' + tokens[4][1:-1].strip()
        cat3, desc3 = tokens[5][1:-1].strip(), 'A_' + tokens[6][1:-1].strip()
        cat4, desc4 = tokens[7][1:-1].strip(), 'A_' + tokens[8][1:-1].strip()

        if icd9.startswith('E'):
            if len(icd9) > 4:
                icd9 = icd9[:4] + '.' + icd9[4:]
        else:
            if len(icd9) > 3:
                icd9 = icd9[:3] + '.' + icd9[3:]
        icd9 = 'D_' + icd9

        if icd9 not in types:
            continue
        icdCode = types[icd9]

        if len(cat4) > 0:
            code1, code2, code3, code4 = types[desc1], types[desc2], types[desc3], types[desc4]
            fiveMap[icdCode] = [icdCode, rootCode, code1, code2, code3, code4]
        elif len(cat3) > 0:
            code1, code2, code3 = types[desc1], types[desc2], types[desc3]
            fourMap[icdCode] = [icdCode, rootCode, code1, code2, code3]
        elif len(cat2) > 0:
            code1, code2 = types[desc1], types[desc2]
            threeMap[icdCode] = [icdCode, rootCode, code1, code2]
        else:
            code1 = types[desc1]
            twoMap[icdCode] = [icdCode, rootCode, code1]
    infd.close()

    # Re-map integer codes
    newFiveMap, newFourMap, newThreeMap, newTwoMap, newOneMap, newTypes = {}, {}, {}, {}, {}, {}
    rtypes = {v: k for k, v in types.items()}

    codeCount = 0
    for dmap, newMap in [(fiveMap, newFiveMap), (fourMap, newFourMap),
                         (threeMap, newThreeMap), (twoMap, newTwoMap),
                         (oneMap, newOneMap)]:
        for icdCode, ancestors in dmap.items():
            newTypes[rtypes[icdCode]] = codeCount
            newMap[codeCount] = [codeCount] + ancestors[1:]
            codeCount += 1

    newSeqs = []
    for patient in seqs:
        newPatient = []
        for visit in patient:
            newVisit = [newTypes[rtypes[code]] for code in visit]
            newPatient.append(newVisit)
        newSeqs.append(newPatient)

    # Ghi kết quả
    pickle.dump(newFiveMap, open(outFile + '.level5.pk', 'wb'))
    pickle.dump(newFourMap, open(outFile + '.level4.pk', 'wb'))
    pickle.dump(newThreeMap, open(outFile + '.level3.pk', 'wb'))
    pickle.dump(newTwoMap, open(outFile + '.level2.pk', 'wb'))
    pickle.dump(newOneMap, open(outFile + '.level1.pk', 'wb'))
    pickle.dump(newTypes, open(outFile + '.types', 'wb'))
    pickle.dump(newSeqs, open(outFile + '.seqs', 'wb'))
