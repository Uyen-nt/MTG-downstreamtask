# build_trees.py
import sys
import pickle

if __name__ == '__main__':
    # -------------------------------
    # 1. Đọc tham số
    # -------------------------------
    infile = sys.argv[1]      # ccs_multi_dx_tool_2015.csv
    seqFile = sys.argv[2]     # mimic.seqs
    typeFile = sys.argv[3]    # mimic.types
    outFile = sys.argv[4]     # mimic3_tree

    print(f"[BUILD TREE] Input: {infile}")
    print(f"             seqs: {seqFile}")
    print(f"             types: {typeFile}")
    print(f"             output: {outFile}.level*.pk")

    # -------------------------------
    # 2. Đọc dữ liệu
    # -------------------------------
    seqs = pickle.load(open(seqFile, 'rb'))
    types = pickle.load(open(typeFile, 'rb'))
    startSet = set(types.keys())
    print(f"Đã đọc: {len(seqs)} bệnh nhân, {len(types)} mã lá")

    # -------------------------------
    # 3. Xây tổ tiên (pass 1)
    # -------------------------------
    hitList = []
    cat1count = cat2count = cat3count = cat4count = 0

    with open(infile, 'r') as infd:
        _ = infd.readline()  # bỏ header
        for line in infd:
            tokens = line.strip().split(',')
            if len(tokens) < 9: continue

            icd9 = tokens[0][1:-1].strip()
            desc1 = 'A_' + tokens[2][1:-1].strip()
            desc2 = 'A_' + tokens[4][1:-1].strip() if len(tokens[4]) > 2 else ""
            desc3 = 'A_' + tokens[6][1:-1].strip() if len(tokens[6]) > 2 else ""
            desc4 = 'A_' + tokens[8][1:-1].strip() if len(tokens[8]) > 2 else ""

            # Chuẩn hóa ICD9
            if icd9.startswith('E'):
                icd9 = icd9[:4] + '.' + icd9[4:] if len(icd9) > 4 else icd9
            else:
                icd9 = icd9[:3] + '.' + icd9[3:] if len(icd9) > 3 else icd9
            icd9 = 'D_' + icd9

            if icd9 in types:
                hitList.append(icd9)

            # Thêm tổ tiên
            if desc1 not in types:
                types[desc1] = len(types)
                cat1count += 1
            if desc2 and desc2 not in types:
                types[desc2] = len(types)
                cat2count += 1
            if desc3 and desc3 not in types:
                types[desc3] = len(types)
                cat3count += 1
            if desc4 and desc4 not in types:
                types[desc4] = len(types)
                cat4count += 1

    # -------------------------------
    # 4. Thêm ROOT
    # -------------------------------
    rootCode = len(types)
    types['A_ROOT'] = rootCode

    print(f"\nTổ tiên:")
    print(f"  L1: {cat1count}, L2: {cat2count}, L3: {cat3count}, L4: {cat4count}")
    print(f"  ROOT: {rootCode}")
    print(f"  Tổng tổ tiên: {cat1count + cat2count + cat3count + cat4count + 1}")
    print(f"  Miss: {len(startSet - set(hitList))}")

    # -------------------------------
    # 5. Xây map theo level (pass 2)
    # -------------------------------
    fiveMap = {}
    fourMap = {}
    threeMap = {}
    twoMap = {}
    oneMap = {}

    # Miss → level1
    missSet = startSet - set(hitList)
    for icd in missSet:
        if icd in types:
            oneMap[types[icd]] = [types[icd], rootCode]

    with open(infile, 'r') as infd:
        _ = infd.readline()
        for line in infd:
            tokens = line.strip().split(',')
            if len(tokens) < 9: continue

            icd9 = tokens[0][1:-1].strip()
            desc1 = 'A_' + tokens[2][1:-1].strip()
            desc2 = 'A_' + tokens[4][1:-1].strip() if len(tokens[4]) > 2 else ""
            desc3 = 'A_' + tokens[6][1:-1].strip() if len(tokens[6]) > 2 else ""
            desc4 = 'A_' + tokens[8][1:-1].strip() if len(tokens[8]) > 2 else ""

            if icd9.startswith('E'):
                icd9 = icd9[:4] + '.' + icd9[4:] if len(icd9) > 4 else icd9
            else:
                icd9 = icd9[:3] + '.' + icd9[3:] if len(icd9) > 3 else icd9
            icd9 = 'D_' + icd9
            if icd9 not in types: continue

            icdCode = types[icd9]
            code1 = types[desc1]

            if desc4:
                fiveMap[icdCode] = [icdCode, rootCode, code1, types[desc2], types[desc3], types[desc4]]
            elif desc3:
                fourMap[icdCode] = [icdCode, rootCode, code1, types[desc2], types[desc3]]
            elif desc2:
                threeMap[icdCode] = [icdCode, rootCode, code1, types[desc2]]
            else:
                twoMap[icdCode] = [icdCode, rootCode, code1]

    # -------------------------------
    # 6. Tái ánh xạ index (rất quan trọng!)
    # -------------------------------
    newFiveMap = {}
    newFourMap = {}
    newThreeMap = {}
    newTwoMap = {}
    newOneMap = {}
    newTypes = {}
    rtypes = {v: k for k, v in types.items()}
    codeCount = 0

    # Duyệt đúng thứ tự: level5 → level1
    for old_map, new_map in [
        (fiveMap, newFiveMap),
        (fourMap, newFourMap),
        (threeMap, newThreeMap),
        (twoMap, newTwoMap),
        (oneMap, newOneMap)
    ]:
        for oldCode, ancestors in old_map.items():
            orig_str = rtypes[oldCode]
            newTypes[orig_str] = codeCount
            new_map[codeCount] = [codeCount] + ancestors[1:]
            codeCount += 1

    print(f"\nTái ánh xạ:")
    print(f"  level5: {len(newFiveMap)}")
    print(f"  level4: {len(newFourMap)}")
    print(f"  level3: {len(newThreeMap)}")
    print(f"  level2: {len(newTwoMap)}")
    print(f"  level1: {len(newOneMap)}")
    print(f"  Tổng mã mới: {len(newTypes)}")

    # -------------------------------
    # 7. Cập nhật seqs
    # -------------------------------
    newSeqs = []
    for patient in seqs:
        newPatient = []
        for visit in patient:
            newVisit = [newTypes[rtypes[code]] for code in visit]
            newPatient.append(newVisit)
        newSeqs.append(newPatient)

    # -------------------------------
    # 8. LƯU FILE
    # -------------------------------
    pickle.dump(newFiveMap, open(outFile + '.level5.pk', 'wb'), -1)
    pickle.dump(newFourMap, open(outFile + '.level4.pk', 'wb'), -1)
    pickle.dump(newThreeMap, open(outFile + '.level3.pk', 'wb'), -1)
    pickle.dump(newTwoMap, open(outFile + '.level2.pk', 'wb'), -1)
    pickle.dump(newOneMap, open(outFile + '.level1.pk', 'wb'), -1)
    pickle.dump(newTypes, open(outFile + '.types', 'wb'), -1)
    pickle.dump(newSeqs, open(outFile + '.seqs', 'wb'), -1)

    print(f"\nSAVED:")
    for i in range(1, 6):
        print(f"  {outFile}.level{i}.pk")
    print(f"  {outFile}.types")
    print(f"  {outFile}.seqs")
    print("DONE")
