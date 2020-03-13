import os
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from numpy.random import rand, randn
from scipy.spatial.distance import pdist, squareform
from matplotlib import pyplot as plt
from shapely import geometry as geo



def star_with_given_circularity(circularity=1, n=7):
    # Spread 2*n points along unit circle
    angles = np.linspace(0, 2*np.pi, 2*n+1)
    a = angles[1]
    x = np.cos(angles)
    y = np.sin(angles)
    xy = np.stack([x,y]).T
    # Calculate inner radius for given circularity
    #    circularity = 4*np.pi * area / perimeter**2
    #    circularity = 4*np.pi * 2*n * (r * np.sin(a) / 2) / (2*n * np.sqrt(r*r + 1 - 2*r*np.cos(a)))**2
    #    circularity = 2*np.pi * r * np.sin(a)) / (2*n * (r*r + 1 - 2*r*np.cos(a)))
    #    r + 1/r = np.pi * np.sin(a) / (n * circularity) + 2*np.cos(a)
    c = np.pi * np.sin(a) / (n * circularity) + 2*np.cos(a)
    if c > 2:
        r = 0.5*(c - np.sqrt(c*c - 4))
        xy[::2,:] *= r
    return xy


def rect_with_given_aspect_and_angle(aspect_ratio, angle):
    xy = np.array([[-1,1], [-1,-1], [1,-1], [1,1], [-1,1]], dtype=float)
    xy[:,1] *= aspect_ratio
    rotation = np.matrix([[np.cos(-angle), np.sin(-angle)], [-np.sin(-angle), np.cos(-angle)]])
    xy = np.dot(rotation, xy.T).T
    return xy



class FourierCurveModel():

    n_parameters = 4*5 # must be uneven number times four
    n_observations = 3
    name = 'fourier-curve'

    def __init__(self):
        self.name = 'fourier-curve'
        self.coeffs_shape = (2, FourierCurveModel.n_parameters//4, 2)
        # Gaussian mixture for generating curve coefficients
        rng = np.random.RandomState(seed=123)
        self.n_components = 5
        self.component_weights = (.5 + rng.rand(self.n_components))
        self.component_weights /= np.sum(self.component_weights)
        self.mus = [.5 * rng.randn(*self.coeffs_shape) for i in range(self.n_components)]
        self.sigmas = [.1 + .2 * rng.rand(*self.coeffs_shape) for i in range(self.n_components)]

    def flatten_coeffs(self, coeffs):
        batch_size = coeffs.shape[0]
        coeffs = coeffs.reshape(batch_size, -1)
        return np.concatenate([coeffs.real, coeffs.imag], axis=1)

    def unflatten_coeffs(self, coeffs):
        batch_size = coeffs.shape[0]
        real, imag = np.split(coeffs, 2, axis=1)
        coeffs = real.astype(np.complex64)
        coeffs.imag = imag
        return coeffs.reshape(batch_size, 2, -1)

    def fourier_coeffs(self, points, n_coeffs=n_parameters//4):
        N = len(points) # Number of points
        M = n_coeffs//2
        M = min(N//2, M) # Number of positive/negative Fourier coefficients
        # Vectorized equation to compute Fourier coefficients
        ms = np.arange(-M, M+1)
        a = np.sum(points[:,:,None] * np.exp(-2*np.pi*1j*ms[None,None,:]*np.arange(N)[:,None,None]/N), axis=0) / N
        return a

    def trace_fourier_curves(self, coeffs, n_points=100):
        # Vectorized equation to compute points along the Fourier curve
        t = np.linspace(0, 1, n_points)
        ms = np.arange(-(coeffs.shape[-1]//2), coeffs.shape[-1]//2 + 1)
        tm = t[:,None] * ms[None,:]
        points = np.sum(coeffs[:,None,:,:] * np.exp(2*np.pi*1j*tm)[None,:,None,:], axis=-1).real
        return points

    def sample_prior(self, n_samples, flat=True):
        samples = []
        for i in range(n_samples):
            c = np.random.choice(self.n_components, p=self.component_weights)
            sample = self.mus[c] + self.sigmas[c] * np.random.randn(*self.coeffs_shape)
            samples.append(sample.astype(np.float32).view(np.complex64))
        samples = np.stack(samples)
        if flat:
            samples = self.flatten_coeffs(samples)
        return samples

    def forward_process(self, x, noise=0.05):
        x = self.unflatten_coeffs(x)
        points = self.trace_fourier_curves(x)
        features = []
        for i in range(len(x)):
            # Find largest diameter of the shape
            d = squareform(pdist(points[i]))
            max_idx = np.unravel_index(d.argmax(), d.shape)
            p0, p1 = points[i,max_idx[0]], points[i,max_idx[1]]
            angle = np.arctan2((p1-p0)[1], (p1-p0)[0])
            max_diameter = d[max_idx]
            # Find largest width orthogonal to diameter
            c, s = np.cos(angle), np.sin(angle)
            rotation = np.matrix([[c, s], [-s, c]])
            p_rotated = np.dot(rotation, points[i].T).T
            min_diameter = np.max(p_rotated[:,1]) - np.min(p_rotated[:,1])
            # Aspect ratio
            aspect_ratio = min_diameter / max_diameter
            # Circularity
            shape = geo.Polygon(points[i])
            circularity = 4*np.pi * shape.area / shape.length**2
            features.append((aspect_ratio, circularity, angle))
        features = np.array(features)
        return features + noise * randn(*features.shape)

    def init_plot(self, y_target=None):
        return plt.figure(figsize=(7,7))

    def update_plot(self, x, y_target=None, n_bold=3, show_forward=True):
        plt.gcf().clear()
        x = self.unflatten_coeffs(np.array(x))
        points = self.trace_fourier_curves(x)
        for i in range(len(points)):
            plt.plot(points[i,:,0], points[i,:,1], c=(0,0,0,min(1,10/len(points))))
            if i >= len(points) - n_bold:
                plt.plot(points[i,:,0], points[i,:,1], c=(0,0,0))
                if show_forward:
                    if y_target is not None:
                        aspect_ratio, circularity, angle = y_target
                        # Visualize circularity
                        star = np.array((4,4)) + .5 * star_with_given_circularity(circularity)
                        plt.plot(star[:,0], star[:,1], c=(0,0,0,.25), lw=1)
                        # Visualize aspect ratio and angle
                        rect = np.array((4,2.5)) + .4 * rect_with_given_aspect_and_angle(aspect_ratio, angle)
                        plt.plot(rect[:,0], rect[:,1], c=(0,0,0,.25), lw=1)
                    # Find largest diameter of the shape
                    d = squareform(pdist(points[i]))
                    max_idx = np.unravel_index(d.argmax(), d.shape)
                    p0, p1 = points[i,max_idx[0]], points[i,max_idx[1]]
                    angle = np.arctan2((p1-p0)[1], (p1-p0)[0])
                    max_diameter = d[max_idx]
                    # Plot
                    d0, d1 = points[i,max_idx[0]], points[i,max_idx[1]]
                    plt.plot([d0[0], d1[0]], [d0[1], d1[1]], c=(0,1,0), ls='-', lw=1)
                    plt.scatter([d0[0], d1[0]], [d0[1], d1[1]], c=[(0,1,0)], s=3, zorder=10)
                    if y_target is not None:
                        # Find largest width orthogonal to diameter
                        c, s = np.cos(angle), np.sin(angle)
                        rotation = np.matrix([[c, s], [-s, c]])
                        p_rotated = np.dot(rotation, points[i].T).T
                        min_diameter = np.max(p_rotated[:,1]) - np.min(p_rotated[:,1])
                        # Aspect ratio & circularity
                        aspect_ratio = min_diameter / max_diameter
                        shape = geo.Polygon(points[i])
                        circularity = 4*np.pi * shape.area / shape.length**2
                        # Visualize circularity
                        star = np.array((4,4)) + .5 * star_with_given_circularity(circularity)
                        plt.plot(star[:,0], star[:,1], c=(0,1,0,.5), ls='-', lw=1)
                        # Visualize aspect ratio and angle
                        rect = np.array((4,2.5)) + .4 * rect_with_given_aspect_and_angle(aspect_ratio, angle)
                        plt.plot(rect[:,0], rect[:,1], c=(0,1,0,.5), ls='-', lw=1)
        plt.axis('equal')
        plt.axis([min(-5, points[:,:,0].min() - 1), max(5, points[:,:,0].max() + 1),
                  min(-5, points[:,:,1].min() - 1), max(5, points[:,:,1].max() + 1)])



class CircleIntersectionModel(FourierCurveModel):

    n_parameters = 4*5 # must be uneven number times four
    n_observations = 3
    name = 'circle-intersection'

    def __init__(self):
        self.name = 'circle-intersection'

    def generate_circle_intersection(self):
        # First circle
        x0, y0, r0 = rand(), rand(), 1 + rand()
        p0 = geo.Point(x0, y0).buffer(r0)
        # Second circle
        r1 = 2 + 2*rand()
        theta = 2*np.pi * rand() # Random angle
        d = (0.6 + 0.2*rand()) * (r0 + r1) # Distance of centers
        x1, y1 = x0 + d * np.sin(theta), y0 + d * np.cos(theta)
        p1 = geo.Point(x1, y1).buffer(r1)
        # Intersect
        shape = p0.intersection(p1)
        # Center with a little noise
        coords = np.array(shape.exterior.coords)
        coords -= coords.mean(axis=0) + 0.2 * randn(1,2)
        return coords

    def sample_prior(self, n_samples, flat=True):
        samples = []
        for i in range(n_samples):
            coords = self.generate_circle_intersection()
            sample = self.fourier_coeffs(coords, n_coeffs=CircleIntersectionModel.n_parameters//4)
            samples.append(sample)
        samples = np.stack(samples)
        if flat:
            samples = self.flatten_coeffs(samples)
        return samples



class PlusShapeModel(FourierCurveModel):

    n_parameters = 4*25 # must be uneven number times four
    n_observations = 3
    name = 'plus-shape'

    def __init__(self):
        self.name = 'plus-shape'

    def densify_polyline(self, coords, max_dist=0.2):
        # Add extra points between consecutive coordinates if they're too far apart
        all = []
        for i in range(len(coords)):
            start = coords[(i+1)%len(coords),:]
            end = coords[i,:]
            dense = np.array([t * start + (1-t) * end
                             for t in np.linspace(0, 1, max(1, int(round(np.max(np.abs(end-start))/max_dist))))])
            all.append(dense)
        return np.concatenate(all)

    def generate_plus_shape(self):
        # Properties of x and y bar
        xlength = 3 + 2 * rand()
        ylength = 3 + 2 * rand()
        xwidth = .5 + 1.5 * rand()
        ywidth = .5 + 1.5 * rand()
        xshift = -1.5 + 3 * rand()
        yshift = -1.5 + 3 * rand()
        # Create bars and compute union
        xbar = geo.box(xshift - xlength/2, -xwidth/2, xshift + xlength/2, xwidth/2)
        ybar = geo.box(-ywidth/2, yshift - ylength/2, ywidth/2, yshift + ylength/2)
        both = xbar.union(ybar)
        coords = np.array(both.exterior.coords[:-1])
        # Add points inbetween, center, rotate and shift randomly
        coords = self.densify_polyline(coords)
        coords -= coords.mean(axis=0)
        angle = 0.5*np.pi * rand()
        rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
        coords = np.dot(coords, rotation)
        coords += 0.5 * randn(1,2)
        return coords

    def sample_prior(self, n_samples, flat=True):
        samples = []
        for i in range(n_samples):
            coords = self.generate_plus_shape()
            sample = self.fourier_coeffs(coords, n_coeffs=PlusShapeModel.n_parameters//4)
            samples.append(sample)
        samples = np.stack(samples)
        if flat:
            samples = self.flatten_coeffs(samples)
        return samples



class FourierCurveDataset(Dataset):

    def __init__(self, model, n, root_dir=None, suffix='', forward=False):
        self.model = model
        self.root_dir = root_dir
        if root_dir is None:
            warnings.warn('FourierCurveDataset: No data directory specified, generated data will not be stored.', Warning)
        self.n = n
        self.suffix = suffix
        if len(suffix) > 0 and not '_' in suffix[:1]:
            suffix = '_' + suffix
        self.forward = forward

        try:
            x = np.load(f'{root_dir}/{self.model.name}_x{suffix}.npy')[:n,...]
        except Exception as e:
            print(f'FourierCurveDataset: Not enough data for model "{self.model.name}" found, generating {n} new samples...')
            x = model.sample_prior(n)
            if root_dir is not None:
                os.makedirs(root_dir, exist_ok=True)
                np.save(f'{root_dir}/{self.model.name}_x{suffix}', x)
        self.x = x
        if forward:
            try:
                y = np.load(f'{root_dir}/{self.model.name}_y{suffix}.npy')[:n,...]
            except Exception as e:
                print(f'FourierCurveDataset: Not enough labels for model "{self.model.name}" found, running forward process on {n} samples...')
                y = []
                if n > 100000:
                    for i in range((n-1)//100000 + 1):
                        print(f'FourierCurveDataset: Forward process chunk {i+1}...')
                        y.append(model.forward_process(x[100000*i : min(n, 100000*(i+1)),...]))
                    y = np.concatenate(y, axis=0)
                else:
                    y = model.forward_process(x)
                print()
                if root_dir is not None:
                    np.save(f'{root_dir}/{self.model.name}_y{suffix}', y)
            self.y = y

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if torch.is_tensor(i):
            i = i.item()

        if self.forward is None:
            return self.x[i]
        else:
            return self.x[i], self.y[i]

    def get_dataloader(self, batch_size):
        return DataLoader(self, batch_size=batch_size, shuffle=True, drop_last=True)



if __name__ == '__main__':
    pass

    model = PlusShapeModel()
    train_data = FourierCurveDataset(model, 100, None, suffix='train', forward=True)
    train_loader = train_data.get_dataloader(25)

    for x,y in train_loader:
        print(x.shape, y.shape)

        fig = model.init_plot()
        model.update_plot(x)
        plt.show()
        break
