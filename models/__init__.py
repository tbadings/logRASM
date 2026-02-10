# Load the benchmarks from the subfiles
from models.linearsystem6D import LinearSystem6D
from .collision_avoidance import CollisionAvoidance
from .drone4D import Drone4D
from .linearsystem import LinearSystem
from .linearsystem4D import LinearSystem4D
from .linearsystem6D import LinearSystem6D
from .pendulum import Pendulum
from .planar_robot import PlanarRobot
from .triple_integrator import TripleIntegrator


def get_model_fun(model_name):
    if model_name == 'LinearSystem':
        envfun = LinearSystem
    elif model_name == 'LinearSystem4D':
        envfun = LinearSystem4D
    elif model_name == 'LinearSystem6D':
        envfun = LinearSystem6D
    elif model_name == 'MyPendulum':
        envfun = Pendulum
    elif model_name == 'CollisionAvoidance':
        envfun = CollisionAvoidance
    elif model_name == 'TripleIntegrator':
        envfun = TripleIntegrator
    elif model_name == 'PlanarRobot':
        envfun = PlanarRobot
    elif model_name == 'Drone4D':
        envfun = Drone4D
    else:
        envfun = False
        assert False, f"Unknown model name: {model_name}"

    return envfun
