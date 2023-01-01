text = input()

list = text.split()
ranks = ["BSO", "CO", "B2", "B2(des)","RSM","S1","USO","3SG", "2SG", "3WO", "2WO", "1WO", "CPT", "MAJ", "LTA", "2LT", "LCP", "PTE", "CPL", "CFC"]
statuses = ["medical", "leave", "off","status", "medical", "appointment", "medical", "leaves", "leaves", "overseas", "leave", "weekend/off", "stay", "out", "personnel", "stay", "in", "personnel", "others"]

print(list)

time = "[ERROR]"
for element in list:
    if element == ":":
        list.remove(element)
    elif element.lower() == "first":
        time = "Morning"
    elif element.lower() == "last":
        time = "Evening"    

for index in range(len(list)):
    try:
        if list[index].lower() == "total" and (list[index+1].lower() == "strength" or list[index+1].lower() == "strength:"):
                list[index:index+2] = ["".join(list[index:index+2])]
    except:
        continue

strength =  0
date = "[ERROR]"
for index in range(len(list)-1):
    if list[index].lower() in ["totalstrength", "totalstrength:"]:
        strength += int(list[index+1])
    if list[index].lower() == "parade":
        date = list[index+1]

new_list = []

for index in range(len(list)):
    if list[index] in ranks or list[index].lower() in statuses:
        new_list.append(list[index])

new_list.append("Others")

for index in range(len(new_list)):
    try:
        if new_list[index].lower() == "medical" and new_list[index+1].lower() == "appointment":
                new_list[index:index+2] = ["".join(new_list[index:index+2])]
    except:
        continue

    try:
        if new_list[index].lower() == "medical" and new_list[index+1].lower() == "leaves":
                new_list[index:index+2] = ["".join(new_list[index:index+2])]
    except:
        continue

    try:
        if new_list[index].lower() == "overseas" and new_list[index+1].lower() == "leave":
                new_list[index:index+2] = ["".join(new_list[index:index+2])]
    except:
        continue

    try:
        if new_list[index].lower() == "stay" and new_list[index+1].lower() == "out" and new_list[index+2].lower() == "personnel":
                new_list[index:index+3] = ["".join(new_list[index:index+3])]
    except:
        continue
    try:
        if new_list[index].lower() == "stay" and new_list[index+1].lower() == "in" and new_list[index+2].lower() == "personnel":
                new_list[index:index+3] = ["".join(new_list[index:index+3])]
    except:
        continue        
    try:
        if new_list[index].lower() == "medical" and new_list[index+1].lower() == "leave":
                new_list[index:index+2] = ["".join(new_list[index:index+2])]
    except:
        continue    

#print(new_list)

joined_statuses = ["medicalleave", "off", "status", "medicalappointment", "medicalleaves", "leaves", "overseasleave", "weekend/off", "stayoutpersonnel", "stayinpersonnel", "others"]
title_index_dict = dict()
title_index_list = []

counter = dict()

for element in joined_statuses:
    counter[element] = 0
counter["course"] = 0

for element in list:
    if element.lower()=="(course)":
        counter["course"]+=1


for index in range(len(new_list)):
    if new_list[index].lower() in joined_statuses:
        title_index_list.append(index)
        title_index_dict[index] = new_list[index].lower()
    if new_list[index].lower() == "(course)":
        counter["course"] += 1


#print(title_index_dict)
#print(title_index_list)



for i in range(len(title_index_list)-1):
    #print(new_list[title_index_list[i]+1 : title_index_list[i+1]])     #for debugging only
    counter[ title_index_dict[title_index_list[i]] ] += len(new_list[title_index_list[i]+1 : title_index_list[i+1]])


#print(f"Unrefined Counter: {counter}")

counter.pop("status")
counter.pop("stayinpersonnel")

for element in counter:
    if element == "off":
        counter["weekend/off"] += counter["off"]
    elif element == "medicalleave":
        counter["medicalleaves"] += counter["medicalleave"]
    elif element == "course":
        counter["others"] = counter["others"] - counter["course"]

counter.pop("off")
counter.pop("medicalleave")
#print(f"Refined Counter: {counter}")

count = 0
for element in counter:
    count += counter[element] 

print(f"""
*HQ {date} {time.upper()}*
{time} Strength: {strength - count}/{strength}

Course: {counter["course"]}
Off/Wkend: {counter["weekend/off"]}
Detention: 0
Hospitalisation: 0
Leave: {counter["leaves"] + counter["overseasleave"]}
MC: {counter["medicalleaves"]}
Others: {counter["others"]}
MA/Dental: {counter["medicalappointment"]}
Stay-Out: {counter["stayoutpersonnel"]}
""")

