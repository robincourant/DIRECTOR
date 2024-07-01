import bpy


def clear_material(material):
    if material.node_tree:
        material.node_tree.links.clear()
        material.node_tree.nodes.clear()


def colored_material_diffuse_BSDF(r, g, b, a=1, roughness=0.127451):
    materials = bpy.data.materials
    material = materials.new(name="body")
    material.use_nodes = True
    clear_material(material)
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    output = nodes.new(type="ShaderNodeOutputMaterial")
    diffuse = nodes.new(type="ShaderNodeBsdfDiffuse")
    diffuse.inputs["Color"].default_value = (r, g, b, a)
    diffuse.inputs["Roughness"].default_value = roughness
    links.new(diffuse.outputs["BSDF"], output.inputs["Surface"])
    return material


DEFAULT_BSDF_SETTINGS = {
    "Subsurface": 0.15,
    "Subsurface Radius": [1.1, 0.2, 0.1],
    "Metallic": 0.3,
    "Specular": 0.5,
    "Specular Tint": 0.5,
    "Roughness": 0.75,
    "Anisotropic": 0.25,
    "Anisotropic Rotation": 0.25,
    "Sheen": 0.75,
    "Sheen Tint": 0.5,
    "Clearcoat": 0.5,
    "Clearcoat Roughness": 0.5,
    "IOR": 1.450,
    "Transmission": 0.1,
    "Transmission Roughness": 0.1,
    "Emission": (0, 0, 0, 1),
    "Emission Strength": 0.0,
    "Alpha": 1.0,
}


def body_material(r, g, b, a=1, name="body", oldrender=True):
    if oldrender:
        material = colored_material_diffuse_BSDF(r, g, b, a=a)
    else:
        materials = bpy.data.materials
        material = materials.new(name=name)
        material.use_nodes = True
        nodes = material.node_tree.nodes
        diffuse = nodes["Principled BSDF"]
        inputs = diffuse.inputs

        settings = DEFAULT_BSDF_SETTINGS.copy()
        settings["Base Color"] = (r, g, b, a)
        settings["Subsurface Color"] = (r, g, b, a)
        settings["Subsurface"] = 0.0

        for setting, val in settings.items():
            inputs[setting].default_value = val

    return material


def colored_material_bsdf(name, **kwargs):
    materials = bpy.data.materials
    material = materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    diffuse = nodes["Principled BSDF"]
    inputs = diffuse.inputs

    settings = DEFAULT_BSDF_SETTINGS.copy()
    for key, val in kwargs.items():
        settings[key] = val

    for setting, val in settings.items():
        inputs[setting].default_value = val

    return material


def floor_mat(name="floor_mat", color=(0.1, 0.1, 0.1, 1), roughness=0.127451):
    return colored_material_diffuse_BSDF(
        color[0], color[1], color[2], a=color[3], roughness=roughness
    )


def plane_mat():
    materials = bpy.data.materials
    material = materials.new(name="plane")
    material.use_nodes = True
    clear_material(material)
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    output = nodes.new(type="ShaderNodeOutputMaterial")
    diffuse = nodes.new(type="ShaderNodeBsdfDiffuse")
    checker = nodes.new(type="ShaderNodeTexChecker")
    checker.inputs["Scale"].default_value = 1024
    checker.inputs["Color1"].default_value = (0.8, 0.8, 0.8, 1)
    checker.inputs["Color2"].default_value = (0.3, 0.3, 0.3, 1)
    links.new(checker.outputs["Color"], diffuse.inputs["Color"])
    links.new(diffuse.outputs["BSDF"], output.inputs["Surface"])
    diffuse.inputs["Roughness"].default_value = 0.127451
    return material


def plane_mat_uni():
    materials = bpy.data.materials
    material = materials.new(name="plane_uni")
    material.use_nodes = True
    clear_material(material)
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    output = nodes.new(type="ShaderNodeOutputMaterial")
    diffuse = nodes.new(type="ShaderNodeBsdfDiffuse")
    diffuse.inputs["Color"].default_value = (0.8, 0.8, 0.8, 1)
    diffuse.inputs["Roughness"].default_value = 0.127451
    links.new(diffuse.outputs["BSDF"], output.inputs["Surface"])
    return material


# ------------------------------------------------------------------------------------- #
# fmt: off
CAM_TO_MATERIAL = {
    0: body_material(0.1450, 0.1450, 0.1450),  # static - black
    1: body_material(0.894, 0.102, 0.110),  # push_in - red
    2: body_material(0.035, 0.415, 0.122),  # pull_out - green
    3: body_material(0.3921, 0.58431, 0.9294),  # boom_bottom - blue
    6: body_material(0.894, 0.894, 0),  # boom_top - yellow
    18: body_material(0.894, 0.102, 0.894),  # trucking_left - magenta
    9: body_material(0.102, 0.894, 0.894),  # trucking_right - cyan
    # ----- #
    12: body_material(0.3921, 0.102, 0.3921),  # right-boom_bottom - purple
    15: body_material(0.3921, 0.3921, 0.035),  # right-boom_top - olive
    21: body_material(0.3921, 0.102, 0.894),  # left-boom_bottom - violet
    24: body_material(0.102, 0.894, 0.3921),  # left-boom_top - teal
    10: body_material(0.894, 0.3921, 0.102),  # right-push_in - orange
    11: body_material(0.894, 0.102, 0.3921),  # right-pull_out - pink
    19: body_material(0.102, 0.3921, 0.3921),  # left-push_in - sea green
    20: body_material(0.3921, 0.894, 0.102),  # left-pull_out - lime
    4: body_material(0.3921, 0.3921, 0.3921),  # boom_bottom-push_in - gray
    5: body_material(0.486, 0.486, 0.486),  # bottom-pull_out - silver
    7: body_material(0.243, 0.243, 0.243),  # boom_top-push_in - dark gray
    8: body_material(0.1450, 0.1450, 0.1450),  # boom_top-pull_out - black
    # ----- #
    13: body_material(0.486, 0.243, 0.435),  # right-boom_bottom-push_in - purple shade
    14: body_material(0.486, 0.435, 0.243),  # right-boom_bottom-pull_out - brown
    16: body_material(0.243, 0.486, 0.243),  # right-boom_top-push_in - dark green
    17: body_material(0.486, 0.243, 0.486),  # right-boom_top-pull_out - purple-pink
    22: body_material(0.243, 0.243, 0.486),  # left-boom_bottom-push_in - indigo
    23: body_material(0.486, 0.486, 0.243),  # left-boom_bottom-pull_out - gold
    25: body_material(0.435, 0.243, 0.243),  # left-boom_top-push_in - brown-red
    26: body_material(0.243, 0.435, 0.486),  # left-boom_top-pull_out - blue-violet
}

MESH_TO_MATERIAL = {
    0: body_material(0.7, 0.7, 0.7),  # static - light gray
    1: body_material(0.9, 0.5, 0.5),  # forward - pastel red
    2: body_material(0.5, 0.9, 0.5),  # backward - pastel green
    3: body_material(0.5, 0.5, 0.9),  # down - pastel blue
    6: body_material(0.9, 0.9, 0.5),  # up - pastel yellow
    18: body_material(0.9, 0.5, 0.9),  # left - pastel magenta
    9: body_material(0.5, 0.9, 0.9),  # right - pastel cyan
    # ----- #
    12: body_material(0.5, 0.5, 0.7),  # right-down - pastel purple
    15: body_material(0.5, 0.7, 0.5),  # right-up - pastel olive
    21: body_material(0.5, 0.5, 0.9),  # left-down - pastel violet
    24: body_material(0.5, 0.9, 0.5),  # left-up - pastel teal
    10: body_material(0.9, 0.5, 0.7),  # right-forward - pastel orange
    11: body_material(0.9, 0.5, 0.7),  # right-backward - pastel pink
    19: body_material(0.5, 0.7, 0.7),  # left-forward - pastel sea green
    20: body_material(0.7, 0.9, 0.5),  # left-backward - pastel lime
    4: body_material(0.5, 0.5, 0.5),  # down-forward - pastel gray
    5: body_material(0.7, 0.7, 0.7),  # down-backward - pastel silver
    7: body_material(0.3, 0.3, 0.3),  # up-forward - pastel dark gray
    8: body_material(0.1, 0.1, 0.1),  # up-backward - pastel black
    # ----- #
    13: body_material(0.7, 0.3, 0.5),  # right-down-forward - pastel purple shade
    14: body_material(0.7, 0.5, 0.3),  # right-down-backward - pastel brown
    16: body_material(0.3, 0.7, 0.3),  # right-up-forward - pastel dark green
    17: body_material(0.7, 0.3, 0.7),  # right-up-backward - pastel purple-pink
    22: body_material(0.3, 0.3, 0.7),  # left-down-forward - pastel indigo
    23: body_material(0.7, 0.7, 0.3),  # left-down-backward - pastel gold
    25: body_material(0.5, 0.3, 0.3),  # left-up-forward - pastel brown-red
    26: body_material(0.3, 0.5, 0.7),  # left-up-backward - pastel blue-violet
}
