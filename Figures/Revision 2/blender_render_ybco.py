import json
import bpy
# https://gifguide2code.com/2017/04/09/python-how-to-code-materials-in-blender-cycles/
def generate_materials(structure):
    colors = structure['colors']
            
    for key, value in colors.items():
        material = bpy.data.materials.get(key)
        if material is None:
            material = bpy.data.materials.new(name=key)
            material.use_nodes = True
            tree = material.node_tree
            
            for node in tree.nodes:
                tree.nodes.remove(node)
                
            material_output = tree.nodes.new(type='ShaderNodeOutputMaterial')
            principled_node = tree.nodes.new(type='ShaderNodeBsdfPrincipled')
            principled_node.location = (-300, 0)
            RGB_node = tree.nodes.new(type='ShaderNodeRGB')
            RGB_node.outputs[0].default_value = value
            RGB_node.location = (-600, 0)
            
            tree.links.new(material_output.inputs['Surface'],
                           principled_node.outputs['BSDF'])
            tree.links.new(principled_node.inputs['Base Color'],
                           RGB_node.outputs['Color'])
            tree.links.new(principled_node.inputs['Base Color'],
                           RGB_node.outputs['Color'])
            
    return

def draw_unit_cell(structure):
        
    for atom in structure['atoms']:
        bpy.ops.mesh.primitive_uv_sphere_add(size=structure['sizes'][atom['element']]*0.3, location=atom['location'])
        bpy.context.active_object.data.materials.append(bpy.data.materials.get(atom['element']))
#        bpy.ops.object.shade_smooth()

with open("C:/Users/pdmurray/Desktop/peyton/Projects/YBCO_Getters/Figures/Revision 2/ybco.json", 'r') as f:
     ybco = json.load(f)

generate_materials(ybco)
draw_unit_cell(ybco)


