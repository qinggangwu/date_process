import xml.etree.ElementTree as ET



xmlpath = '/home/wqg/data/maxvision_data/annotations.xml'
tree = ET.ElementTree(file=xmlpath)
root = tree.getroot()
child_root = root[1:30]

# print(child_root.getroot())
print(len(root))
print(root.tag,root.attrib)
print('child')
for child_of_image in child_root:
    # child = child_of_root
    # print(child_of_image.tag, child_of_image.attrib)
    print(child_of_image.attrib)
    for child_of_point in child_of_image:
        # print(child_of_point.tag, child_of_point.attrib)
        print(child_of_point.attrib)
        for child in child_of_point:
            print(child.attrib['name'] ,child.text)
        # print(child_of_root.tag, child_of_root.attrib)



# for elem in tree.iter(tag='image'):
#     e = tree.iter(tag='point')
#     print(e)
#     # print(e.tag, e.attrib)
#     print(elem.tag, elem.attrib)

