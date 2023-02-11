import meshio
import numpy as np
from tqdm import tqdm

class SaveParaview:
    def __init__(self):
        self.xmesh = np.linspace(0, 1, 11)
        self.ymesh = np.linspace(0, 1, 11)
        self.tmesh = np.linspace(0, 1, 11)
        self.filename = "asd.xdmf"
        self.fields = {}

    def create_mesh(self):
        nx, ny, nt = len(self.xmesh), len(self.ymesh), len(self.tmesh)
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

    def create_point_data(self):
        # values.shape = (nt, nx, ny)
        self.points_timed_data = []
        for k, tk in enumerate(self.tmesh):
            newpointdata = {}
            for name, field in self.fields.items():
                newpointdata[name] = field[k].flatten()
            self.points_timed_data.append(newpointdata)

    def save(self):
        self.create_mesh()
        self.create_point_data()
        print("  # Saving inside function")
        with meshio.xdmf.TimeSeriesWriter(self.filename) as writer:
            writer.write_points_cells(self.points, self.cells)
            for k, tk in enumerate(tqdm(self.tmesh)):
                writer.write_data(tk, point_data=self.points_timed_data[k])
            