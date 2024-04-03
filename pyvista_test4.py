import pyvista as pv
import numpy as np


def clip_parallel_to_camera(point):
    """Clip all actors in the scene with a plane defined by two picked points
    and the camera position"""

    if len(p._picked_points) == 0:
        # store the first point and its actor
        p._picked_points.append(point)
        p._picked_point_actors = [p.actors["_picked_point"]]
    else:
        # calculate the normal of the cutting plane.
        # The plane is defined by the
        # two picked points and the position of the camera Use one of the points
        # as origin and as normal the vector perpendicular to both the vectors
        # defined by the camera and the picked points.
        origin = p._picked_points[0]
        cameraPosition = p.camera.GetPosition()
        v0 = origin - cameraPosition
        v1 = point - cameraPosition
        normal = np.cross(v0, v1)

        # remove actors of  points since they are not relevant anymore
        p._picked_point_actors.append(p.actors["_picked_point"])
        for actor in p._picked_point_actors:
            p.remove_actor(actor)
        # make a copy of current actors. Required since we are going to add more
        # actors as we iterate.
        current_actors = list(p.actors.values())
        for actor in current_actors:
            # skip annotations
            if type(actor) == pv.Actor:
                # Apply the filter only to visible datasets
                if actor.visibility:
                    mesh = actor.mapper.GetInputDataObject(0, 0)
                    clipped = mesh.clip(origin=origin, normal=normal)
                    if clipped.n_points != 0:
                        p.add_mesh(clipped)
                        # hide original dataset
                        actor.visibility = False
        # we are done, reset the state
        p.disable_picking()
        p._picked_points = []
        p._picked_point_actors = []


mesh1 = pv.Cube()
mesh2 = pv.Cube().translate(xyz=[2, 0, 0])
mesh3 = pv.Cube().translate(xyz=[4, 0, 0])

p = pv.Plotter()

p.add_mesh(mesh1)
p.add_mesh(mesh2)
p.add_mesh(mesh3)


def toggle():
    p.enable_point_picking(
        callback=clip_parallel_to_camera,
        pickable_window=True,
        show_message="Pick two points for clipping",
    )
    p._picked_points = []


p.add_key_event("c", toggle)

p.show()