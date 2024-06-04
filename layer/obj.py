import numpy as np


class Obj:
    def __init__(self, vertex, face, init_index, vertex_file, face_file, file_path, output_inst_color, one_building_inst_color):
        self.vertex = vertex
        self.face = face
        self.init_index = init_index
        self.vertex_file = vertex_file
        self.face_file = face_file
        self.file_path = file_path
        self.output_inst_color = output_inst_color
        self.one_building_inst_color = one_building_inst_color

        # self.write_vertex()
        # self.write_face()

    def write_vertex(self):
        with open(self.vertex_file, "a+") as f:
            for vertex in self.vertex:
                vertex_data = "v " + " ".join(map(str, vertex[:3])) + " "

                if self.output_inst_color:
                    vertex_data += " ".join(map(str, self.one_building_inst_color[:3])) + " "

                vertex_data += "\n"
                f.write(vertex_data)

    def write_face(self):
        with open(self.face_file, "a+") as f:
            for face in self.face:
                face_data = "f " + " ".join(map(str, face)) + "\n"
                f.write(face_data)

    def write_info(self):
        with open(self.file_path, "a+") as f:
            for vertex in self.vertex:
                vertex_data = "v " + " ".join(map(str, vertex[:3])) + " "
                if self.output_inst_color:
                    vertex_data += " ".join(map(str, self.one_building_inst_color[:3])) + " "
                vertex_data += "\n"
                f.write(vertex_data)
            for face in self.face:
                face_data = "f " + " ".join(map(str, face)) + "\n"
                f.write(face_data)
