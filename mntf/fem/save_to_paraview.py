import meshio
import numpy as np
from tqdm import tqdm

class SaveParaview:
    def __init__(self):
        self.xmesh = np.linspace(0, 1, 11)
        self.ymesh = np.linspace(0, 1, 11)
        self.filename = "asd.xdmf"
        self.fields = {}
        self.writer = None

    def create_mesh(self):
        nx, ny = len(self.xmesh), len(self.ymesh)
        self.points = []
        quadcells = []
        indexs = {}
        k = 0
        for i, xi in enumerate(self.xmesh):
            for j, yj in enumerate(self.ymesh):
                indexs[xi, yj] = k
                k += 1
                self.points.append((xi, yj))
        for i in range(nx-1):
            for j in range(ny-1):
                newquadcell = []
                newquadcell.append(indexs[self.xmesh[i], self.ymesh[j]])
                newquadcell.append(indexs[self.xmesh[i+1], self.ymesh[j]])
                newquadcell.append(indexs[self.xmesh[i+1], self.ymesh[j+1]])
                newquadcell.append(indexs[self.xmesh[i], self.ymesh[j+1]])
                quadcells.append(newquadcell)
        self.cells = [ ("quad", quadcells) ]


    def open_writer(self):
        self.create_mesh()
        print("    # Opening xdmf writer")
        self.writer = meshio.xdmf.TimeSeriesWriter(self.filename)
        self.writer.__enter__()
        self.writer.write_points_cells(self.points, self.cells)

    def close_writer(self):
        print("    # Closing xdmf writer")
        self.writer.__exit__()
        self.writer = None

    def write_at_time(self, tk: float, name: str, field: np.ndarray):
        if self.writer is None:
            raise ValueError("The file must be opened to write on it")
        point_data = {name: field.flatten()}
        self.writer.write_data(tk, point_data=point_data)

                

            