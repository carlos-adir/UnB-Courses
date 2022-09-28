import numpy as np
from typing import Tuple, Iterable, Optional
from matplotlib import pyplot as plt

def compute_distance(x: Iterable[float], y: Iterable[float], A: float, B: float):
    """
    Given n points, it will return the distance from each point to a ellipse of form
    (x/A)^2 + (y/B)^2 = 1

    To do it, we say that
        x = A * cos(t)
        y = B * sin(t)
    with 0 <= t <= np.pi/2
    Then we want to minimize
        D^2 = (x-x0)^2 + (y-y0)^2 = (A*cos(t)-x0)^2 + (B*sin(t)-y0)^2
        d(D^2)/dt = -2*A*sin(t)*(A*cos(t)-x0) + 2*B*cos(t)*(B*sin(t)-y0)
        f(t) = A*x0 * sin(t) - B*y0 * cos(t) - cos(t)*sin(t) * (A^2-B^2)
        df(t) = A*x0 * cos(t) + B*y0 * sin(t) - cos(2*t)*(A^2-B^2)
    To solve, for each point we use bissection method and after newtons method to find theta.
    We see that
        f(0) = -B*y0 < 0
        f(np.pi/2) = A*x0 > 0
    """
    n = len(x)
    d2 = np.zeros(n)
    C2 = A**2 - B**2
    ts = np.zeros(n)
    for i, (xi, yi) in enumerate(zip(x, y)):
        ta, tb = 0, np.pi/2
        fa = -B*yi
        fb = A*xi
        for j in range(5):  # 5 iterations for bissection method
            tm = 0.5*(ta+tb)
            cos, sin = np.cos(tm), np.sin(tm)
            fm = A*xi*sin - B*yi*cos - C2*cos*sin
            if fa*fm < 0:
                tb = tm
                fb = fm
            else:
                ta = tm
                fa = fm
        for j in range(5):  # Newtons iteration
            cos, sin, cos2 = np.cos(tm), np.sin(tm), np.cos(2*tm)
            fm = A*xi*sin - B*yi*cos - C2*cos*sin
            dfm = A*xi*cos + B*yi*sin - cos2*C2
            tm -= fm/dfm
            if tm > np.pi/2:
                tm = np.pi/2
            elif tm < 0:
                tm = 0
        ts[i] = tm
    for i, (xi, yi) in enumerate(zip(x, y)):
        tm = ts[i]
        cos, sin = np.cos(tm), np.sin(tm)
        dsq = (A*cos-xi)**2 + (B*sin-yi)**2
        d0 = (A-xi)**2 + yi**2
        d1 = xi**2 + (B-yi)**2
        d2[i] = min(d0, d1, dsq)
    return np.sqrt(d2)


def rotate_points(x: Iterable[float], y: Iterable[float], angle: float):
    n = len(x)
    R = np.array([[np.cos(angle), np.sin(angle)],
                  [-np.sin(angle), np.cos(angle)]])
    xn, yn = np.zeros(n), np.zeros(n)
    for i in range(n):
        xn[i], yn[i] = R @ (x[i], y[i])
    return xn, yn    

class Ellipse:

    def __init__(self):
        self.__coeffs = None
        self.__sizeaxis = (1, 1)
        self.__angle = 0
        self.__center = (0, 0)
        self.update_coeffs()

    def fit(self, x: Iterable[float], y: Iterable[float]):
        """
        Fit the coefficients a,b,c,d,e,f, representing an ellipse described by
        the formula F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0 to the provided
        arrays of data points x=[x1, x2, ..., xn] and y=[y1, y2, ..., yn].

        Based on the algorithm of Halir and Flusser, "Numerically stable direct
        least squares fitting of ellipses'.
        """
        x = np.array(x)
        y = np.array(y)
        D1 = np.vstack([x**2, x*y, y**2]).T
        D2 = np.vstack([x, y, np.ones(len(x))]).T
        S1 = D1.T @ D1
        S2 = D1.T @ D2
        S3 = D2.T @ D2
        T = -np.linalg.inv(S3) @ S2.T
        M = S1 + S2 @ T
        C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
        M = np.linalg.inv(C) @ M
        eigval, eigvec = np.linalg.eig(M)
        con = 4 * eigvec[0]* eigvec[2] - eigvec[1]**2
        ak = eigvec[:, np.nonzero(con > 0)[0]]
        coeffs = np.concatenate((ak, T @ ak)).ravel()
        coeffs /= np.sqrt(4*coeffs[0]*coeffs[2]-coeffs[1]**2)
        self.__coeffs = coeffs
        self.update_polar()

    def __compute_polar(self):
        """
        Convert the cartesian conic coefficients, (a, b, c, d, e, f), to the
        ellipse parameters, where F(x, y) = ax^2 + bxy + cy^2 + dx + ey + f = 0.
        The returned parameters are x0, y0, ap, bp, e, phi, where (x0, y0) is the
        ellipse centre; (ap, bp) are the semi-major and semi-minor axes,
        respectively; e is the eccentricity; and phi is the rotation of the semi-
        major axis from the x-axis.

        """
        # We use the formulas from https://mathworld.wolfram.com/Ellipse.html
        # which assumes a cartesian form ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0.
        # Therefore, rename and scale b, d and f appropriately.
        if self.coeffs is None:
            raise ValueError("You must fit the model first")
        coeffs = np.copy(self.coeffs)
        a = coeffs[0]
        b = coeffs[1] / 2
        c = coeffs[2]
        d = coeffs[3] / 2
        f = coeffs[4] / 2
        g = coeffs[5]

        den = b**2 - a*c
        if den > 0:
            raise ValueError('coeffs do not represent an ellipse: b^2 - 4ac must'
                            ' be negative!')

        # The location of the ellipse centre.
        x0, y0 = (c*d - b*f) / den, (a*f - b*d) / den

        num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
        fac = np.sqrt((a - c)**2 + 4*b**2)
        # The semi-major and semi-minor axis lengths (these are not sorted).
        ap = np.sqrt(num / den / (fac - a - c))
        bp = np.sqrt(num / den / (-fac - a - c))

        # Sort the semi-major and semi-minor axis lengths but keep track of
        # the original relative magnitudes of width and height.
        width_gt_height = True
        if ap < bp:
            width_gt_height = False
            ap, bp = bp, ap

        # The eccentricity.
        r = (bp/ap)**2
        if r > 1:
            r = 1/r
        e = np.sqrt(1 - r)

        # The angle of anticlockwise rotation of the major-axis from x-axis.
        if b == 0:
            phi = 0 if a < c else np.pi/2
        else:
            phi = np.arctan((2.*b) / (a - c)) / 2
            if a > c:
                phi += np.pi/2
        if not width_gt_height:
            # Ensure that phi is the angle to rotate to the semi-major axis.
            phi += np.pi/2
        phi = phi % np.pi
        self.__center = (x0, y0)
        self.__sizeaxis = ap, bp
        self.__angle = phi
        self.update_coeffs()
        
    @property
    def coeffs(self):
        return np.copy(self.__coeffs)
    
    @property
    def center(self):
        return np.copy(self.__center)

    @property
    def amplitude(self):
        if self.coeffs is None:
            raise ValueError("Cannot get it! coeffs were not set")
        a, b, c, d, e, f = self.coeffs
        xmed, ymed = self.center
        xamp = np.sqrt(xmed**2 - 4*c*f + e**2)
        yamp = np.sqrt(ymed**2 - 4*a*f + d**2)
        return np.array([xamp, yamp])

    @property
    def axis(self):
        if self.__sizeaxis is None:
            self.__compute_polar()
        return np.copy(self.__sizeaxis)

    @property
    def angle(self):
        return self.__angle


    @coeffs.setter
    def coeffs(self, value: Tuple[float]):
        self.__coeffs = np.copy(value)
        self.update_polar()

    @center.setter
    def center(self, value: Tuple[float]):
        self.__center = np.array(value)
        self.update_coeffs()

    @angle.setter
    def angle(self, value: float):
        self.__angle = float(value)
        self.update_coeffs()

    @axis.setter
    def axis(self, value: Tuple[float]):
        self.__sizeaxis = np.array(value)
        self.update_coeffs()

    def distance(self, x: Iterable[float], y: Iterable[float]):
        x, y = np.array(x, dtype="float64"), np.array(y, dtype="float64")
        xc, yc = self.center
        phi = self.angle
        A, B = self.axis
        x -= xc
        y -= yc
        xn, yn = np.abs(rotate_points(x, y, phi))
        distances = compute_distance(xn, yn, A, B)
        return distances

    def sample(self, npts: int=129, tmin: float=0, tmax: float=2*np.pi):
        """
        Return npts points on the ellipse described by the params = x0, y0, ap,
        bp, e, phi for values of the parametric variable t between tmin and tmax.
        """
        center = self.center
        A, B = self.axis
        phi = self.angle
        # A grid of the parametric variable, t.
        t = np.linspace(tmin, tmax, npts)
        x = center[0] + A * np.cos(t) * np.cos(phi) - B * np.sin(t) * np.sin(phi)
        y = center[1] + A * np.cos(t) * np.sin(phi) + B * np.sin(t) * np.cos(phi)
        return x, y

    def update_coeffs(self):
        xc, yc = self.center
        A, B = self.axis
        phi = self.angle
        sin, cos = -np.sin(phi), np.cos(phi)
        a = A**2 * sin**2 + B**2 * cos**2
        b = (A**2 - B**2)* 2*sin*cos
        c = A**2 * cos**2 + B**2 * sin**2
        d = -2*A**2*sin*(xc*sin + yc*cos)
        d += -2*B**2*cos*(xc*cos - yc*sin)
        e = -2*A**2*cos*(xc*sin + yc*cos)
        e += 2*B**2*sin*(xc*cos - yc*sin)
        f = A**2*(xc*sin + yc*cos)**2
        f += B**2*(xc*cos - yc*sin)**2
        f -= A**2*B**2
        denom = np.sqrt(4*a*c - b**2)
        self.__coeffs = np.array([a, b, c, d, e, f])/denom
        
    def update_polar(self):
        self.__compute_polar()


    def get_point_ymax(self):
        a, b, c, d, e, f = self.coeffs
        ymed = self.center[1]
        delta = ymed**2 - 4*a*f + d**2
        ymax = ymed + np.sqrt(delta)
        xref = -(d+b*ymax)/(2*a)
        return (xref, ymax)

    def get_point_xmax(self):
        a, b, c, d, e, f = self.coeffs
        xmed = self.center[0]
        delta = xmed**2 - 4*c*f + e**2
        xmax = xmed + np.sqrt(delta)
        yref = -(e+b*xmax)/(2*c)
        return (xmax, yref)

    def get_point_ymin(self):
        a, b, c, d, e, f = self.coeffs
        ymed = self.center[1]
        delta = ymed**2 - 4*a*f + d**2
        ymin = ymed - np.sqrt(delta)
        xref = -(d+b*ymin)/(2*a)
        return (xref, ymin)

    def get_point_xmin(self):
        a, b, c, d, e, f = self.coeffs
        xmed = self.center[0]
        delta = xmed**2 - 4*c*f + e**2
        xmin = xmed - np.sqrt(delta)
        yref = -(e+b*xmin)/(2*c)
        return (xmin, yref)


def test_circle():
    ellipse = Ellipse()
    ellipse.coeffs = 1, 0, 1, 0, 0, -1
    np.testing.assert_almost_equal(ellipse.center, (0, 0))
    np.testing.assert_almost_equal(ellipse.axis, (1, 1))
    assert (ellipse.angle % (np.pi/2)) == 0

def test_ellipsescales():
    ellipse = Ellipse()
    ntests = 100
    for i in range(ntests):
        A, B = 2*np.random.rand(2)+1
        if A < B:
            A, B = B, A
        ellipse.coeffs = (1/A**2), 0, (1/B**2), 0, 0, -1
        np.testing.assert_almost_equal(ellipse.center, (0, 0))
        np.testing.assert_almost_equal(ellipse.axis, (A, B))
        assert ellipse.angle == 0

def test_ellipsetranslated():
    ellipse = Ellipse()
    A, B = 1, 0.5
    ellipse.coeffs = 1/A**2, 0, 1/B**2, 0, 0, -1
    ntests = 100
    for i in range(ntests):
        xc, yc = 2*np.random.rand(2)-1
        ellipse.center = (xc, yc)
        np.testing.assert_almost_equal(ellipse.center, (xc, yc))
        np.testing.assert_almost_equal(ellipse.axis, (A, B))
        assert ellipse.angle == 0        

def test_ellipserotated():
    ellipse = Ellipse()
    A, B = 1, 0.5
    ellipse.coeffs = 1/(A**2), 0, 1/(B**2), 0, 0, -1
    ntests = 100
    for i in range(ntests):
        angle = np.pi*np.random.rand()/2
        ellipse.angle = angle
        np.testing.assert_almost_equal(ellipse.center, (0, 0))
        np.testing.assert_almost_equal(ellipse.axis, (A, B))
        assert abs(ellipse.angle - angle) < 1e-9
        

def test_distancecircle():
    ellipse = Ellipse()
    ntests = 100
    npoints = 100
    for i in range(ntests):
        radius = 9*np.random.rand()+1
        xc, yc = 5*np.random.rand(2) - 2.5
        angle = np.pi*np.random.rand()
        ellipse.center = (xc, yc)
        ellipse.axis = (radius, radius)
        ellipse.angle = angle
        x0s, y0s = 50*np.random.rand(2, npoints)-25
        good_distances = np.sqrt( (x0s-xc)**2 + (y0s-yc)**2) - radius
        good_distances = np.abs(good_distances)
        test_distances = ellipse.distance(x0s, y0s)
        np.testing.assert_almost_equal(test_distances, good_distances)

def test_distanceelipse():
    ellipse = Ellipse()
    ntests = 100
    npoints = 100
    for i in range(ntests):
        A, B = 5*np.random.rand(2) + 1
        if A < B:
            A, B = B, A
        xc, yc = 5*np.random.rand(2) - 2.5
        angle = np.pi*np.random.rand()
        ellipse.center = (xc, yc)
        ellipse.axis = (A, B)
        ellipse.angle = angle
        x0s, y0s = ellipse.sample(npoints)
        good_distances = np.zeros(npoints)
        test_distances = ellipse.distance(x0s, y0s)
        np.testing.assert_array_almost_equal(test_distances, good_distances)
        
        
def test_boxenclap():
    ellipse = Ellipse()
    ntests = 100
    npoints = 100
    for i in range(ntests):
        A, B = 5*np.random.rand(2) + 1
        if A < B:
            A, B = B, A
        xc, yc = 5*np.random.rand(2) - 2.5
        angle = np.pi*np.random.rand()
        ellipse.center = (xc, yc)
        ellipse.axis = (A, B)
        ellipse.angle = angle
        x0s, y0s = ellipse.sample(npoints)
        xmed, ymed = ellipse.center
        xamp, yamp = ellipse.amplitude

        assert np.all( (xmed-xamp <= x0s) * (x0s <= xmed+xamp) )
        assert np.all( (ymed-yamp <= y0s) * (y0s <= ymed+yamp) )

def test_fitvals():
    ntests = 100
    ellipse = Ellipse()
    newellipse = Ellipse()
    for i in range(100):
        xc, yc = 10*np.random.rand(2)-5
        A, B = 5*np.random.rand(2)+1
        if A < B:
            A, B = B, A
        angle = np.pi*np.random.rand()
        ellipse.center = (xc, yc)
        ellipse.axis = (A, B)
        ellipse.angle = angle
        xvals, yvals = ellipse.sample()
        newellipse.fit(xvals, yvals)
        np.testing.assert_allclose(ellipse.coeffs, newellipse.coeffs)
        np.testing.assert_allclose(newellipse.center, (xc, yc))
        np.testing.assert_allclose(newellipse.axis, (A, B))
        assert abs(newellipse.angle - angle) < 1e-9


def main():
    test_circle()
    test_ellipsescales()
    test_ellipsetranslated()
    test_ellipserotated()
    test_distancecircle()
    test_distanceelipse()
    test_boxenclap()
    test_fitvals()

if __name__ == "__main__":
    main()
