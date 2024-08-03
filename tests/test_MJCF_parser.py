import xml.etree.ElementTree as ET

# 加载和解析 XML 文件
tree = ET.parse('/home/jiechu/Data/TinyRobotBench/manipulate/assets/object/catapult/catapult.xml')
root = tree.getroot()

# 打印根元素的标签和属性
print(f"Root element: {root.tag}, attributes: {root.attrib}")

# 遍历 XML 文件中的所有 'material' 元素
for material in root.findall(".//material"):
    name = material.get('name')
    rgba = material.get('rgba')
    print(f"Material Name: {name}, Color: {rgba}")

# 查找特定的 'body' 元素并访问其子元素
for body in root.findall(".//body[@name='catapult']"):
    print(f"Body: {body.get('name')}")
    for geom in body.findall('geom'):
        print(f"  Geom Type: {geom.get('type')}, Size: {geom.get('size')}")

# 如果你需要修改 XML 树
for material in root.findall(".//material"):
    if material.get('name') == 'material_0':
        material.set('rgba', '1.0 1.0 1.0 1')  # 更改颜色属性

# 保存修改后的 XML 到新文件
tree.write('modified_mujoco_file.xml')