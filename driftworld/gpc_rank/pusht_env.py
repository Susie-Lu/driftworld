"""
Push-T env, adapted from https://github.com/han20192019/gpc_code/blob/main/gpc_rank_evaluation/pusht_env.py
"""
from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import collections
import torch
import torchvision
from torchvision.transforms import v2
from PIL import Image

# env import
import gymnasium as gym
from gymnasium import spaces
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.space_debug_draw_options import SpaceDebugColor
from pymunk.vec2d import Vec2d
import shapely.geometry as sg
import cv2
import skimage.transform as st
import os
import yaml

#@markdown ### **Environment**
#@markdown Defines a PyMunk-based Push-T environment `PushTEnv`.
#@markdown And it's subclass `PushTImageEnv`.
#@markdown
#@markdown **Goal**: push the gray T-block into the green area.
#@markdown
#@markdown Adapted from [Implicit Behavior Cloning](https://implicitbc.github.io/)

positive_y_is_up: bool = False
"""Make increasing values of y point upwards.

When True::

    y
    ^
    |      . (3, 3)
    |
    |   . (2, 2)
    |
    +------ > x

When False::

    +------ > x
    |
    |   . (2, 2)
    |
    |      . (3, 3)
    v
    y

"""

def to_pygame(p: Tuple[float, float], surface: pygame.Surface) -> Tuple[int, int]:
    """Convenience method to convert pymunk coordinates to pygame surface
    local coordinates.

    Note that in case positive_y_is_up is False, this function wont actually do
    anything except converting the point to integers.
    """
    if positive_y_is_up:
        return round(p[0]), surface.get_height() - round(p[1])
    else:
        return round(p[0]), round(p[1])


def light_color(color: SpaceDebugColor):
    color = np.minimum(1.2 * np.float32([color.r, color.g, color.b, color.a]), np.float32([255]))
    color = SpaceDebugColor(r=color[0], g=color[1], b=color[2], a=color[3])
    return color

class DrawOptions(pymunk.SpaceDebugDrawOptions):
    def __init__(self, surface: pygame.Surface) -> None:
        """Draw a pymunk.Space on a pygame.Surface object.

        Typical usage::

        >>> import pymunk
        >>> surface = pygame.Surface((10,10))
        >>> space = pymunk.Space()
        >>> options = pymunk.pygame_util.DrawOptions(surface)
        >>> space.debug_draw(options)

        You can control the color of a shape by setting shape.color to the color
        you want it drawn in::

        >>> c = pymunk.Circle(None, 10)
        >>> c.color = pygame.Color("pink")

        See pygame_util.demo.py for a full example

        Since pygame uses a coordiante system where y points down (in contrast
        to many other cases), you either have to make the physics simulation
        with Pymunk also behave in that way, or flip everything when you draw.

        The easiest is probably to just make the simulation behave the same
        way as Pygame does. In that way all coordinates used are in the same
        orientation and easy to reason about::

        >>> space = pymunk.Space()
        >>> space.gravity = (0, -1000)
        >>> body = pymunk.Body()
        >>> body.position = (0, 0) # will be positioned in the top left corner
        >>> space.debug_draw(options)

        To flip the drawing its possible to set the module property
        :py:data:`positive_y_is_up` to True. Then the pygame drawing will flip
        the simulation upside down before drawing::

        >>> positive_y_is_up = True
        >>> body = pymunk.Body()
        >>> body.position = (0, 0)
        >>> # Body will be position in bottom left corner

        :Parameters:
                surface : pygame.Surface
                    Surface that the objects will be drawn on
        """
        self.surface = surface
        super(DrawOptions, self).__init__()

    def draw_circle(
        self,
        pos: Vec2d,
        angle: float,
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        p = to_pygame(pos, self.surface)

        pygame.draw.circle(self.surface, fill_color.as_int(), p, round(radius), 0)
        pygame.draw.circle(self.surface, light_color(fill_color).as_int(), p, round(radius-4), 0)

        circle_edge = pos + Vec2d(radius, 0).rotated(angle)
        p2 = to_pygame(circle_edge, self.surface)
        line_r = 2 if radius > 20 else 1
        # pygame.draw.lines(self.surface, outline_color.as_int(), False, [p, p2], line_r)

    def draw_segment(self, a: Vec2d, b: Vec2d, color: SpaceDebugColor) -> None:
        p1 = to_pygame(a, self.surface)
        p2 = to_pygame(b, self.surface)

        pygame.draw.aalines(self.surface, color.as_int(), False, [p1, p2])

    def draw_fat_segment(
        self,
        a: Tuple[float, float],
        b: Tuple[float, float],
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        p1 = to_pygame(a, self.surface)
        p2 = to_pygame(b, self.surface)

        r = round(max(1, radius * 2))
        pygame.draw.lines(self.surface, fill_color.as_int(), False, [p1, p2], r)
        if r > 2:
            orthog = [abs(p2[1] - p1[1]), abs(p2[0] - p1[0])]
            if orthog[0] == 0 and orthog[1] == 0:
                return
            scale = radius / (orthog[0] * orthog[0] + orthog[1] * orthog[1]) ** 0.5
            orthog[0] = round(orthog[0] * scale)
            orthog[1] = round(orthog[1] * scale)
            points = [
                (p1[0] - orthog[0], p1[1] - orthog[1]),
                (p1[0] + orthog[0], p1[1] + orthog[1]),
                (p2[0] + orthog[0], p2[1] + orthog[1]),
                (p2[0] - orthog[0], p2[1] - orthog[1]),
            ]
            pygame.draw.polygon(self.surface, fill_color.as_int(), points)
            pygame.draw.circle(
                self.surface,
                fill_color.as_int(),
                (round(p1[0]), round(p1[1])),
                round(radius),
            )
            pygame.draw.circle(
                self.surface,
                fill_color.as_int(),
                (round(p2[0]), round(p2[1])),
                round(radius),
            )

    def draw_polygon(
        self,
        verts: Sequence[Tuple[float, float]],
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        ps = [to_pygame(v, self.surface) for v in verts]
        ps += [ps[0]]

        radius = 2
        pygame.draw.polygon(self.surface, light_color(fill_color).as_int(), ps)

        if radius > 0:
            for i in range(len(verts)):
                a = verts[i]
                b = verts[(i + 1) % len(verts)]
                self.draw_fat_segment(a, b, radius, fill_color, fill_color)

    def draw_dot(
        self, size: float, pos: Tuple[float, float], color: SpaceDebugColor
    ) -> None:
        p = to_pygame(pos, self.surface)
        pygame.draw.circle(self.surface, color.as_int(), p, round(size), 0)


def pymunk_to_shapely(body, shapes):
    geoms = list()
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
        else:
            raise RuntimeError(f'Unsupported shape type {type(shape)}')
    geom = sg.MultiPolygon(geoms)
    return geom

# env
class PushTEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    reward_range = (0., 1.)

    def __init__(self,
            legacy=False,
            block_cog=None, damping=None,
            render_action=True,
            render_size=96,
            reset_to_state=None,
            domain_filename=None     # NEW CODE
        ):
        self._seed = None
        self.seed()
        self.window_size = ws = 512  # The size of the PyGame window
        self.render_size = render_size
        self.sim_hz = 100
        # Local controller params.
        self.k_p, self.k_v = 100, 20    # PD control.z
        self.control_hz = self.metadata['video.frames_per_second']
        # legcay set_state for data compatiblity
        self.legacy = legacy

        # NEW CODE: domain index for loading environment settings (colors, misleading locations etc.)
        self.domain_filename = domain_filename
        with open("./domains_yaml/{}.yml".format(self.domain_filename), 'r') as stream:
            data_loaded = yaml.safe_load(stream)
        self.block_color = data_loaded["block_color"]
        self.target_color = data_loaded["target_color"]
        self.bg_color = data_loaded["bg_color"]
        self.obstacle_color = data_loaded["obstacle_color"]
        self.block_scale = data_loaded["block_scale"]
        self.num_mislead = data_loaded["num_mislead"]
        self.num_obstacle = data_loaded["num_obstacle"]
        self.wrong_scale = self.block_scale * 2
        self.object = data_loaded["object"]

        # agent_pos, block_pos, block_angle
        self.observation_space = spaces.Box(
            low=np.array([0,0,0,0,0], dtype=np.float64),
            high=np.array([ws,ws,ws,ws,np.pi*2], dtype=np.float64),
            shape=(5,),
            dtype=np.float64
        )

        # positional goal for agent
        self.action_space = spaces.Box(
            low=np.array([0,0], dtype=np.float64),
            high=np.array([ws,ws], dtype=np.float64),
            shape=(2,),
            dtype=np.float64
        )

        self.block_cog = block_cog
        self.damping = damping
        self.render_action = render_action

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.screen = None

        self.space = None
        self.teleop = None
        self.render_buffer = None
        self.latest_action = None
        self.reset_to_state = reset_to_state

    def reset(self):
        seed = self._seed
        self._setup()
        if self.block_cog is not None:
            self.block.center_of_gravity = self.block_cog
        if self.damping is not None:
            self.space.damping = self.damping

        # use legacy RandomState for compatiblity
        state = self.reset_to_state
        # NEW CODE: add randomization for target locations and expand range to maximum possible region
        rs = np.random.RandomState(seed=seed)
        if state is None:
            state = np.array([
                rs.uniform(20, 490), rs.uniform(20, 490),
                rs.uniform(15+3*self.block_scale, 498-3*self.block_scale),
                rs.uniform(15+3*self.block_scale, 498-3*self.block_scale),
                rs.randn() * 2 * np.pi - np.pi,
                rs.uniform(15+3*self.block_scale, 498-3*self.block_scale),
                rs.uniform(15+3*self.block_scale, 498-3*self.block_scale),
                rs.randn() * 2 * np.pi - np.pi
                ])

            while np.linalg.norm(state[:2]-state[2:4]) < 3*self.block_scale:
                state[0] = rs.uniform(20, 490)
                state[1] = rs.uniform(20, 490)            

        if self.object == "circle":
            state[4] = 0
            state[7] = 0
        self._set_state(state)

        # NEW CODE
        geoms_block = []
        geoms_goal = []
        goal_body = self._get_goal_pose_body(self.goal_pose)
        for shape in self.block.shapes:
            if isinstance(shape, pymunk.shapes.Poly):
                verts_block = [self.block.local_to_world(v) for v in shape.get_vertices()]
                verts_block += [verts_block[0]]
                geoms_block.append(sg.Polygon(verts_block))

                verts_goal = [goal_body.local_to_world(v) for v in shape.get_vertices()]
                verts_goal += [verts_goal[0]]
                geoms_goal.append(sg.Polygon(verts_goal))
            elif isinstance(shape, pymunk.shapes.Circle):
                geoms_block=[self.block.position, shape.radius]
                geoms_goal=[self.goal_pose[:2], shape.radius]

        obstacle_centers = []
        self.obstacle_rng = rs.rand(self.num_obstacle)
        i = 0
        while i < self.num_obstacle:
            center = rs.uniform(5+self.wrong_scale, 506-self.wrong_scale, 2).tolist()
            # if obstacles are overlapped with the agent point
            if sg.Point(center).distance(sg.Point(state[:2])) < self.wrong_scale+15:
                continue

            # if obstacles are overlapped with each other
            is_overlap = False
            for prev_center in obstacle_centers:
                if sg.Point(center).distance(sg.Point(prev_center)) < (2*self.wrong_scale):
                    is_overlap = True
                    break
            if is_overlap:
                continue

            if self.object != "circle" and self.obstacle_rng[i]<0.5: # circle
                center_circle = center
                # if obstacles are overlapped with the block
                dist = sg.Point(center_circle).distance(geoms_block[0])
                for geom in geoms_block:
                    dist = min(dist, sg.Point(center_circle).distance(geom))
                if dist < self.wrong_scale:
                    continue
                # if obstacles are overlapped with the goal
                dist = sg.Point(center_circle).distance(geoms_goal[0])
                for geom in geoms_goal:
                    dist = min(dist, sg.Point(center_circle).distance(geom))
                if dist < self.wrong_scale:
                    continue
                       
                obstacle = self.add_obstacle_circle(center_circle, self.wrong_scale, self.obstacle_color)
                obstacle_centers.append(center_circle)
                i += 1
            else:   # triangle
                cx = center[0]
                cy = center[1]
                rot_angle = rs.uniform(0, np.pi/3*2)
                tri_x1 = self.wrong_scale * np.sin(rot_angle)
                tri_y1 = self.wrong_scale * np.cos(rot_angle)
                tri_x2 = self.wrong_scale * np.sin(rot_angle+np.pi/3*2)
                tri_y2 = self.wrong_scale * np.cos(rot_angle+np.pi/3*2)
                tri_x3 = self.wrong_scale * np.sin(rot_angle+np.pi/3*4)
                tri_y3 = self.wrong_scale * np.cos(rot_angle+np.pi/3*4)
                tri_points = [[tri_x1, tri_y1], [tri_x2, tri_y2], [tri_x3, tri_y3]]
                abs_tri_points = [[tri_x1+cx, tri_y1+cy], 
                                  [tri_x2+cx, tri_y2+cy], 
                                  [tri_x3+cx, tri_y3+cy]]
                tri_geom = sg.Polygon(abs_tri_points)
                
                is_overlap = False

                if self.object=="circle":
                    dist1 = sg.Point(geoms_block[0]).distance(tri_geom)
                    dist2 = sg.Point(geoms_goal[0]).distance(tri_geom)
                    if dist1 < geoms_block[1] or dist2 < geoms_goal[1]:
                        is_overlap = True
                else:
                    # if obstacles are overlapped with the block
                    for geom in geoms_block:
                        if geom.intersects(tri_geom):
                            is_overlap = True
                            break
                    # if obstacles are overlapped with the goal
                    for geom in geoms_goal:
                        if geom.intersects(tri_geom):
                            is_overlap = True
                            break
                
                if is_overlap:
                    continue
                
                obstacle = self.add_obstacle_triangle([cx, cy], tri_points, self.obstacle_color)
                obstacle_centers.append([cx, cy])
                i += 1

        self.mislead_circles = []
        self.mislead_triangles = []
        self.mislead_rng = rs.rand(self.num_mislead)
        i = 0

        while i < self.num_mislead:
            if self.object != "circle" and self.mislead_rng[i]<0.5: # circle
                center_circle = rs.uniform(5+self.wrong_scale, 506-self.wrong_scale, 2)
                dist = geoms_goal[0].distance(sg.Point(center_circle))
                for geom in geoms_goal:
                    dist = min(dist, geom.distance(sg.Point(center_circle)))

                if dist < self.wrong_scale:
                    continue
                else:
                    self.mislead_circles.append(center_circle)
                    i += 1
            else:   # triangle
                cx = rs.uniform(5+self.wrong_scale, 506-self.wrong_scale)
                cy = rs.uniform(5+self.wrong_scale, 506-self.wrong_scale)
                rot_angle = rs.uniform(0, np.pi/3*2)
                tri_x1 = cx + self.wrong_scale * np.sin(rot_angle)
                tri_y1 = cy + self.wrong_scale * np.cos(rot_angle)
                tri_x2 = cx + self.wrong_scale * np.sin(rot_angle+np.pi/3*2)
                tri_y2 = cy + self.wrong_scale * np.cos(rot_angle+np.pi/3*2)
                tri_x3 = cx + self.wrong_scale * np.sin(rot_angle+np.pi/3*4)
                tri_y3 = cy + self.wrong_scale * np.cos(rot_angle+np.pi/3*4)
                tri_points = [(tri_x1, tri_y1), (tri_x2, tri_y2), (tri_x3, tri_y3)]
                tri_geom = sg.Polygon(tri_points)

                is_overlap = False

                if self.object=="circle":
                    dist = sg.Point(geoms_goal[0]).distance(tri_geom)
                    if dist < geoms_goal[1]:
                        is_overlap = True
                else:
                    for geom in geoms_goal:
                        if geom.intersects(tri_geom):
                            is_overlap = True
                            break
                if is_overlap:
                    continue
                else:
                    self.mislead_triangles.append(tri_points)
                    i += 1

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        dt = 1.0 / self.sim_hz
        self.n_contact_points = 0
        n_steps = self.sim_hz // self.control_hz
        if action is not None:
            self.latest_action = action
            for i in range(n_steps):
                # Step PD control.
                # self.agent.velocity = self.k_p * (act - self.agent.position)    # P control works too.
                acceleration = self.k_p * (action - self.agent.position) + self.k_v * (Vec2d(0, 0) - self.agent.velocity)
                self.agent.velocity += acceleration * dt

                # Step physics.
                self.space.step(dt)

        # compute reward
        if self.object == "circle":
            c1 = self.block.position
            c2 = self.goal_pose[:2]
            r = list(self.block.shapes)[0].radius
            d = np.linalg.norm(c1-c2)
            if d == 0:
                coverage = 1
            elif d > (2*r):
                coverage = 0
            else:
                theta = np.arccos(0.5*d/r)*2
                coverage = (theta - np.sin(theta))/(2*np.pi)*2
        else:
            goal_body = self._get_goal_pose_body(self.goal_pose)
            goal_geom = pymunk_to_shapely(goal_body, self.block.shapes)
            block_geom = pymunk_to_shapely(self.block, self.block.shapes)

            intersection_area = goal_geom.intersection(block_geom).area
            goal_area = goal_geom.area
            coverage = intersection_area / goal_area
        reward = np.clip(coverage / self.success_threshold, 0, 1)
        done = coverage > self.success_threshold
        terminated = done
        truncated = done

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self, mode):
        return self._render_frame(mode)

    def teleop_agent(self):
        TeleopAgent = collections.namedtuple('TeleopAgent', ['act'])
        def act(obs):
            act = None
            mouse_position = pymunk.pygame_util.from_pygame(Vec2d(*pygame.mouse.get_pos()), self.screen)
            if self.teleop or (mouse_position - self.agent.position).length < 30:
                self.teleop = True
                act = mouse_position
            return act
        return TeleopAgent(act)

    def _get_obs(self):
        if self.object == "circle":
            angle = 0
        else:
            angle = self.block.angle

        obs = np.array(
            tuple(self.agent.position) \
            + tuple(self.block.position) \
            + (angle % (2 * np.pi),))
        return obs

    def _get_goal_pose_body(self, pose):
        mass = 2
        inertia = pymunk.moment_for_box(mass, (50, 100))
        body = pymunk.Body(mass, inertia)
        # preserving the legacy assignment order for compatibility
        # the order here dosn't matter somehow, maybe because CoM is aligned with body origin
        body.position = pose[:2].tolist()
        body.angle = pose[2]
        return body

    def _get_info(self):
        n_steps = self.sim_hz // self.control_hz
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))
        if self.object == "circle":
            angle = 0
        else:
            angle = self.block.angle

        info = {
            'pos_agent': np.array(self.agent.position),
            'vel_agent': np.array(self.agent.velocity),
            'block_pose': np.array(list(self.block.position) + [angle]),
            'goal_pose': self.goal_pose,
            'n_contacts': n_contact_points_per_step}
        return info

    def _render_frame(self, mode, out_size=None):

        # out_size lets callers request a different output resolution (e.g. a high-res
        # ground-truth render); defaults to self.render_size so existing callers are unchanged.
        size = out_size if out_size is not None else self.render_size

        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        # NEW CODE:
        canvas.fill(pygame.Color(self.bg_color))
        self.screen = canvas

        draw_options = DrawOptions(canvas)

        # Draw goal pose.
        if self.object == "circle":
            r = list(self.block.shapes)[0].radius
            pygame.draw.circle(canvas, self.goal_color, self.goal_pose[:2], r)
        else:
            goal_body = self._get_goal_pose_body(self.goal_pose)
            for shape in self.block.shapes:
                goal_points = [pymunk.pygame_util.to_pygame(goal_body.local_to_world(v), draw_options.surface) for v in shape.get_vertices()]
                goal_points += [goal_points[0]]
                pygame.draw.polygon(canvas, self.goal_color, goal_points)

        for i in range(len(self.mislead_circles)):
            # Draw circle location  
            pygame.draw.circle(canvas, self.goal_color, self.mislead_circles[i], self.wrong_scale)

        for i in range(len(self.mislead_triangles)):
            # Draw triangle location 
            pygame.draw.polygon(canvas, self.goal_color, self.mislead_triangles[i])

        # Draw agent and block.
        self.space.debug_draw(draw_options)

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # the clock is aleady ticked during in step for "human"


        img = np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        img = cv2.resize(img, (size, size))
        if self.render_action:
            if self.render_action and (self.latest_action is not None):
                action = np.array(self.latest_action)
                coord = (action / 512 * size).astype(np.int32)
                marker_size = max(1, int(8/96*size))
                thickness = max(1, int(1/96*size))
                cv2.drawMarker(img, coord,
                    color=(255,0,0), markerType=cv2.MARKER_CROSS,
                    markerSize=marker_size, thickness=thickness)
        return img


    def _render_frame_target_only(self, mode):

        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        # NEW CODE:
        canvas.fill(pygame.Color(self.bg_color))
        self.screen = canvas

        draw_options = DrawOptions(canvas)

        # Draw goal pose.
        if self.object == "circle":
            r = list(self.block.shapes)[0].radius
            pygame.draw.circle(canvas, self.goal_color, self.goal_pose[:2], r)
        else:
            goal_body = self._get_goal_pose_body(self.goal_pose)
            for shape in self.block.shapes:
                goal_points = [pymunk.pygame_util.to_pygame(goal_body.local_to_world(v), draw_options.surface) for v in shape.get_vertices()]
                goal_points += [goal_points[0]]
                pygame.draw.polygon(canvas, self.goal_color, goal_points)

       

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # the clock is aleady ticked during in step for "human"


        img = np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        img = cv2.resize(img, (self.render_size, self.render_size))
        if self.render_action:
            if self.render_action and (self.latest_action is not None):
                action = np.array(self.latest_action)
                coord = (action / 512 * 96).astype(np.int32)
                marker_size = int(8/96*self.render_size)
                thickness = int(1/96*self.render_size)
                cv2.drawMarker(img, coord,
                    color=(255,0,0), markerType=cv2.MARKER_CROSS,
                    markerSize=marker_size, thickness=thickness)
        return img


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0,25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    def _handle_collision(self, arbiter, space, data):
        self.n_contact_points += len(arbiter.contact_point_set.points)

    def _set_state(self, state):
        if isinstance(state, np.ndarray):
            state = state.tolist()
        pos_agent = state[:2]
        pos_block = state[2:4]
        rot_block = state[4]
        self.agent.position = pos_agent
        self.goal_pose = np.array(state[5:8])
        # setting angle rotates with respect to center of mass
        # therefore will modify the geometric position
        # if not the same as CoM
        # therefore should be modified first.
        if self.legacy:
            # for compatiblity with legacy data
            self.block.position = pos_block
            if self.object != "circle":
                self.block.angle = rot_block
        else:
            if self.object != "circle":
                self.block.angle = rot_block
            self.block.position = pos_block

        # Run physics to take effect
        self.space.step(1.0 / self.sim_hz)

    def _set_state_local(self, state_local):
        agent_pos_local = state_local[:2]
        block_pose_local = state_local[2:]
        tf_img_obj = st.AffineTransform(
            translation=self.goal_pose[:2],
            rotation=self.goal_pose[2])
        tf_obj_new = st.AffineTransform(
            translation=block_pose_local[:2],
            rotation=block_pose_local[2]
        )
        tf_img_new = st.AffineTransform(
            matrix=tf_img_obj.params @ tf_obj_new.params
        )
        agent_pos_new = tf_img_new(agent_pos_local)
        new_state = np.array(
            list(agent_pos_new[0]) + list(tf_img_new.translation) \
                + [tf_img_new.rotation])
        self._set_state(new_state)
        return new_state

    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0
        self.teleop = False
        self.render_buffer = list()

        # Add walls.
        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2)
        ]
        self.space.add(*walls)

        # Add agent, block, and goal zone.
        self.agent = self.add_circle((256, 400), 15)
        self.block = self.add_object((256, 300), 0, self.object, self.block_scale, self.block_color)
        self.goal_color = pygame.Color(self.target_color)
        self.goal_pose = np.array([256,256,np.pi/4])  # x, y, theta (in radians)

        # Add collision handeling
        # self.collision_handeler = self.space.add_collision_handler(0, 0)
        # self.collision_handeler.post_solve = self._handle_collision
        # self.n_contact_points = 0
        self.space.on_collision(0, 0, post_solve=self._handle_collision)
        self.n_contact_points = 0

        self.max_score = 50 * 100
        self.success_threshold = 0.95    # 95% coverage.

    def _add_segment(self, a, b, radius):
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        shape.color = pygame.Color('LightGray')    # https://htmlcolorcodes.com/color-names
        return shape

    def add_circle(self, position, radius):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color('RoyalBlue')
        self.space.add(body, shape)
        return body

    def add_obstacle_circle(self, position, radius, color = 'RoyalBlue'):
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = position
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color(color)
        self.space.add(body, shape)
        return body

    def add_obstacle_triangle(self, position, vertices, color = 'RoyalBlue'):
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = position
        shape = pymunk.Poly(body, vertices)
        shape.color = pygame.Color(color)
        self.space.add(body, shape)
        return body

    def add_box(self, position, height, width):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (height, width))
        body = pymunk.Body(mass, inertia)
        body.position = position
        shape = pymunk.Poly.create_box(body, (height, width))
        shape.color = pygame.Color('LightSlateGray')
        self.space.add(body, shape)
        return body

    def add_object(self, position, angle, object:str, scale=30, color='LightSlateGray'):
       
        object_list = ["circle"]
        letter_list = ["T", "H", "V", "A", "D", "R"]
        if (not (object.upper() in letter_list)) and (not (object in object_list)):
            raise Exception("Object should be in the list: {}".format(letter_list+object_list))
        
        if object == "T":
            return self.add_tee(position, angle, scale, color)
        elif object == "H":
            return self.add_H(position, angle, scale, color)
        elif object == "V":
            return self.add_V(position, angle, scale, color)
        elif object == "A":
            return self.add_A(position, angle, scale, color)
        elif object == "D":
            return self.add_D(position, angle, scale, color)
        elif object == "R":
            return self.add_R(position, angle, scale, color)
        elif object == "circle":
            return self.add_ball(position, scale, color)
        else:
            raise

    def add_tee(self, position, angle, scale=30, color='LightSlateGray', mask=pymunk.ShapeFilter.ALL_MASKS()):
        mass = 1
        length = 4
        mass1 = mass*2*length/(length*2-1)
        mass2 = mass*2*(length-1)/(length*2-1)
        CoM_offset = (length+2)*scale/4
        vertices1 = [(-length*scale/2, scale-CoM_offset),
                                 ( length*scale/2, scale-CoM_offset),
                                 ( length*scale/2, -CoM_offset),
                                 (-length*scale/2, -CoM_offset)]
        inertia1 = pymunk.moment_for_poly(mass1, vertices=vertices1)
        vertices2 = [(-scale/2, scale-CoM_offset),
                                 (-scale/2, length*scale-CoM_offset),
                                 ( scale/2, length*scale-CoM_offset),
                                 ( scale/2, scale-CoM_offset)]
        inertia2 = pymunk.moment_for_poly(mass2, vertices=vertices2)

        length_CoM = length*scale/4
        body = pymunk.Body(mass*2, inertia1 + inertia2 + length_CoM*length_CoM*mass*2) # parallel axis theorem
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity) / 2
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1, shape2)
        return body

    def add_ball(self, position, scale=30, color='LightSlateGray', mask=pymunk.ShapeFilter.ALL_MASKS()):
        mass = 1
        length = 4
        radius = length*scale/2

        inertia = pymunk.moment_for_circle(mass*2, inner_radius=0, outer_radius=radius)

        body = pymunk.Body(mass*2, inertia)
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color(color)
        shape.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = shape.center_of_gravity
        body.position = position
        body.friction = 1
        self.space.add(body, shape)
        return body

    def add_H(self, position, angle, scale=30, color='LightSlateGray', mask=pymunk.ShapeFilter.ALL_MASKS()):
        mass = 1
        length = 4
        mass1 = mass*2*length/(3*length-2)
        mass2 = mass*2*(length-2)/(3*length-2)

        vertices1 = [(-length*scale/2,     -length*scale/2),
                     (-length*scale/2,      length*scale/2),
                     (-(length-2)*scale/2,  length*scale/2),
                     (-(length-2)*scale/2, -length*scale/2)]
        inertia1 = pymunk.moment_for_poly(mass1, vertices=vertices1)

        vertices2 = [(length*scale/2,     -length*scale/2),
                     (length*scale/2,      length*scale/2),
                     ((length-2)*scale/2,  length*scale/2),
                     ((length-2)*scale/2, -length*scale/2)]
        inertia2 = pymunk.moment_for_poly(mass1, vertices=vertices2)

        vertices3 = [(-(length-2)*scale/2, -scale/2),
                     (-(length-2)*scale/2,  scale/2),
                     ( (length-2)*scale/2,  scale/2),
                     ( (length-2)*scale/2, -scale/2)]
        inertia3 = pymunk.moment_for_poly(mass2, vertices=vertices3)

        length_CoM = (length-1)*scale/2
        body = pymunk.Body(mass*2, inertia1 + inertia2 + inertia3 + length_CoM*length_CoM*mass1*2) # parallel axis theorem
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape3 = pymunk.Poly(body, vertices3)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape3.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        shape3.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity + shape3.center_of_gravity) / 2
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1, shape2, shape3)
        return body
    
    def add_V(self, position, angle, scale=30, color='LightSlateGray', mask=pymunk.ShapeFilter.ALL_MASKS()):
        mass = 1
        length = 4
        CoM_offset = length*scale*(6*length*length-12*length+5)/(3*(length-1)*(4*length-5))

        vertices1 = [(0, -CoM_offset),
                     (-scale/2, -CoM_offset),
                     (-length*scale/2,     length*scale-CoM_offset),
                     (-(length-2)*scale/2, length*scale-CoM_offset),
                     (0, length*scale/(length-1)-CoM_offset)]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)

        vertices2 = [(0, -CoM_offset),
                     (scale/2, -CoM_offset),
                     (length*scale/2,     length*scale-CoM_offset),
                     ((length-2)*scale/2, length*scale-CoM_offset),
                     (0, length*scale/(length-1)-CoM_offset)]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices2)

        length_CoM = (6*length*length-12*length+7)/(6*(4*length-5))*scale

        body = pymunk.Body(mass*2, inertia1 + inertia2 + length_CoM*length_CoM*mass*2) # parallel axis theorem
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity) / 2
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1, shape2)
        return body
    
    def add_A(self, position, angle, scale=30, color='LightSlateGray', mask=pymunk.ShapeFilter.ALL_MASKS()):
        mass = 1
        length = 4
        x1 = (length*length-3*length+1)*scale/(2*length)
        y1 = (length-1)*scale
        x2 = (2*length*length-7*length+3)*scale/(4*length)
        y2 = (length-1.5)*scale
        y_12 = (y1+y2)/2
        area1 = length*scale*scale*(4*length-5)/(4*(length-1))
        area2 = (2*x1+2*x2)*scale/4
        mass1 = mass*2*area1/(area1*2+area2)
        mass2 = mass*2*area2/(area1*2+area2)

        CoM_offset = length*scale*(6*length*length-12*length+5)/(3*(length-1)*(4*length-5))
        CoM_offset = (CoM_offset*area1*2 + y_12*area2)/(area1*2+area2)

        vertices1 = [(0, -CoM_offset),
                     (-scale/2, -CoM_offset),
                     (-length*scale/2,     length*scale-CoM_offset),
                     (-(length-2)*scale/2, length*scale-CoM_offset),
                     (0, length*scale/(length-1)-CoM_offset)]
        inertia1 = pymunk.moment_for_poly(mass1, vertices=vertices1)

        vertices2 = [(0, -CoM_offset),
                     (scale/2, -CoM_offset),
                     (length*scale/2,     length*scale-CoM_offset),
                     ((length-2)*scale/2, length*scale-CoM_offset),
                     (0, length*scale/(length-1)-CoM_offset)]
        inertia2 = pymunk.moment_for_poly(mass1, vertices=vertices2)

        vertices3 = [( x1, y1-CoM_offset), 
                     ( x2, y2-CoM_offset), 
                     (-x2, y2-CoM_offset),
                     (-x1, y1-CoM_offset)]
        inertia3 = pymunk.moment_for_poly(mass2, vertices=vertices3)

        length_CoM = (6*length*length-12*length+7)/(6*(4*length-5))*scale

        body = pymunk.Body(mass*2, inertia1 + inertia2 + inertia3 + length_CoM*length_CoM*mass1*2) # parallel axis theorem
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape3 = pymunk.Poly(body, vertices3)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape3.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        shape3.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity + shape3.center_of_gravity) / 2
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1, shape2, shape3)
        return body
    
    def arc_to_poly(self, center, radius, start_angle=np.pi/2, tot_angle=np.pi, num_points=10):
        poly_points = []

        if tot_angle>(2*np.pi) or tot_angle<0:
            raise Exception("A valid arc should be in range (0, 2*pi]!")
        if num_points<3:
            raise Exception("num_points should be at least 3!")
        
        angle = tot_angle/(num_points-1)
        for i in range(num_points):
            cur_angle = start_angle+i*angle
            cur_x = center[0] - radius*np.cos(cur_angle)
            cur_y = center[1] + radius*np.sin(cur_angle)
            poly_points.append((cur_x, cur_y))
        
        return poly_points

    
    def add_D(self, position, angle, scale=30, color='LightSlateGray', mask=pymunk.ShapeFilter.ALL_MASKS()):
        mass = 1

        mass1 = mass*2*4/(6+1.5*np.pi)
        mass2 = mass*2/(6+1.5*np.pi)
        mass3 = mass*2*1.5*np.pi/(6+1.5*np.pi)

        CoM_offset = scale*(5+3*np.pi)/(6+1.5*np.pi)

        vertices1 = [(-CoM_offset, -2*scale),
                     (-CoM_offset,  2*scale),
                     ( scale-CoM_offset,  2*scale),
                     ( scale-CoM_offset, -2*scale)]
        inertia1 = pymunk.moment_for_poly(mass1, vertices=vertices1)

        vertices2 = [(scale-CoM_offset,   scale),
                     (scale-CoM_offset,   2*scale),
                     (2*scale-CoM_offset, 2*scale),
                     (2*scale-CoM_offset, scale)]
        inertia2 = pymunk.moment_for_poly(mass2, vertices=vertices2)

        vertices3 = [(scale-CoM_offset,   -scale),
                     (scale-CoM_offset,   -2*scale),
                     (2*scale-CoM_offset, -2*scale),
                     (2*scale-CoM_offset, -scale)]
        inertia3 = pymunk.moment_for_poly(mass2, vertices=vertices3)

        # polygons to estimate semi-circle ring
        num_points = 6
        semi_circle1 = self.arc_to_poly(center=(2*scale-CoM_offset, 0), radius=scale, start_angle=np.pi/2, num_points=num_points)
        semi_circle2 = self.arc_to_poly(center=(2*scale-CoM_offset, 0), radius=scale*2, start_angle=np.pi/2, num_points=num_points)
        
        poly_list = []
        tot_poly_inertia = 0
        for i in range(num_points-1):
            idx1 = i
            idx2 = i+1
            poly_vertices = [semi_circle1[idx1], semi_circle1[idx2], semi_circle2[idx2], semi_circle2[idx1]]
            poly_inertia = pymunk.moment_for_poly(mass3/(num_points-1), vertices=poly_vertices)
            poly_list.append(poly_vertices)
            tot_poly_inertia += poly_inertia

        length_CoM = CoM_offset-0.5*scale
        total_inertia = inertia1 + inertia2 + inertia3 + tot_poly_inertia + \
                length_CoM*length_CoM*mass1 + 2*2.25*scale*scale*mass2 + (2*scale-CoM_offset)*(2*scale-CoM_offset)*mass3
        body = pymunk.Body(mass*2, total_inertia) # parallel axis theorem
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape3 = pymunk.Poly(body, vertices3)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape3.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        shape3.filter = pymunk.ShapeFilter(mask=mask)

        all_shapes = [shape1, shape2, shape3]
        for poly in poly_list:
            poly_shape = pymunk.Poly(body, poly)
            poly_shape.color = pygame.Color(color)
            poly_shape.filter = pymunk.ShapeFilter(mask=mask)
            all_shapes.append(poly_shape)
        
        body.center_of_gravity = (0, 0)
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, *all_shapes)
        return body
    
    def add_R(self, position, angle, scale=30, color='LightSlateGray', mask=pymunk.ShapeFilter.ALL_MASKS()):
        mass = 1

        area1 = 3
        area2 = 0.5
        area3 = 1.125
        area4 = np.pi/2
        tot_area = area1+area2*2+area3+area4
        
        mass1 = mass*2*area1/tot_area
        mass2 = mass*2*area2/tot_area
        mass3 = mass*2*area3/tot_area
        mass4 = mass*2*area4/tot_area

        CoM_offset_x = -(405+64*np.pi)*scale/(328+32*np.pi)
        CoM_offset_y = (307+44*np.pi)*scale/(164+16*np.pi)

        vec1 = [-0.625*scale-CoM_offset_x, 2*scale-CoM_offset_y]
        vec2 = [-1.5*scale-CoM_offset_x, 3.75*scale-CoM_offset_y]
        vec3 = [-1.5*scale-CoM_offset_x, 1.75*scale-CoM_offset_y]
        vec4 = [-2.625*scale-CoM_offset_x, 0.75*scale-CoM_offset_y]
        vec5 = [-2*scale-CoM_offset_x, 2.75*scale-CoM_offset_y]

        vertices1 = [(-0.25*scale-CoM_offset_x,        -CoM_offset_y),
                     (-0.25*scale-CoM_offset_x, 4*scale-CoM_offset_y),
                     (     -scale-CoM_offset_x, 4*scale-CoM_offset_y),
                     (     -scale-CoM_offset_x,        -CoM_offset_y)]
        inertia1 = pymunk.moment_for_poly(mass1, vertices=vertices1)

        vertices2 = [(     -scale-CoM_offset_x, 3.5*scale-CoM_offset_y),
                     (     -scale-CoM_offset_x,   4*scale-CoM_offset_y),
                     (   -2*scale-CoM_offset_x,   4*scale-CoM_offset_y),
                     (   -2*scale-CoM_offset_x, 3.5*scale-CoM_offset_y)]
        inertia2 = pymunk.moment_for_poly(mass2, vertices=vertices2)

        vertices3 = [(     -scale-CoM_offset_x, 1.5*scale-CoM_offset_y),
                     (     -scale-CoM_offset_x,   2*scale-CoM_offset_y),
                     (   -2*scale-CoM_offset_x,   2*scale-CoM_offset_y),
                     (   -2*scale-CoM_offset_x, 1.5*scale-CoM_offset_y)]
        inertia3 = pymunk.moment_for_poly(mass2, vertices=vertices3)

        vertices4 = [(-1.25*scale-CoM_offset_x, 1.5*scale-CoM_offset_y),
                     (   -2*scale-CoM_offset_x, 1.5*scale-CoM_offset_y),
                     (   -4*scale-CoM_offset_x,          -CoM_offset_y),
                     (-3.25*scale-CoM_offset_x,          -CoM_offset_y)]
        inertia4 = pymunk.moment_for_poly(mass3, vertices=vertices4)

        # polygons to estimate semi-circle ring
        num_points = 10
        semi_circle1 = self.arc_to_poly(center=(-2*scale-CoM_offset_x, 2.75*scale-CoM_offset_y), radius=0.75*scale, start_angle=-np.pi/2, num_points=num_points)
        semi_circle2 = self.arc_to_poly(center=(-2*scale-CoM_offset_x, 2.75*scale-CoM_offset_y), radius=1.25*scale, start_angle=-np.pi/2, num_points=num_points)
        
        poly_list = []
        tot_poly_inertia = 0
        for i in range(num_points-1):
            idx1 = i
            idx2 = i+1
            poly_vertices = [semi_circle1[idx1], semi_circle1[idx2], semi_circle2[idx2], semi_circle2[idx1]]
            poly_inertia = pymunk.moment_for_poly(mass4/(num_points-1), vertices=poly_vertices)
            poly_list.append(poly_vertices)
            tot_poly_inertia += poly_inertia

        total_inertia = inertia1 + inertia2 + inertia3 + inertia4 + tot_poly_inertia + \
                        (vec1[0]*vec1[0]+vec1[1]*vec1[1])*mass1 + \
                        (vec2[0]*vec2[0]+vec2[1]+vec2[1]+vec3[0]*vec3[0]+vec3[1]+vec3[1])*mass2 + \
                        (vec4[0]*vec4[0]+vec4[1]*vec4[1])*mass3 +\
                        (vec5[0]*vec5[0]+vec5[1]*vec5[1])*mass4
        body = pymunk.Body(mass*2, total_inertia) # parallel axis theorem
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape3 = pymunk.Poly(body, vertices3)
        shape4 = pymunk.Poly(body, vertices4)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape3.color = pygame.Color(color)
        shape4.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        shape3.filter = pymunk.ShapeFilter(mask=mask)
        shape4.filter = pymunk.ShapeFilter(mask=mask)

        all_shapes = [shape1, shape2, shape3, shape4]
        for poly in poly_list:
            poly_shape = pymunk.Poly(body, poly)
            poly_shape.color = pygame.Color(color)
            poly_shape.filter = pymunk.ShapeFilter(mask=mask)
            all_shapes.append(poly_shape)
        
        body.center_of_gravity = (0, 0)
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, *all_shapes)
        return body

class PushTImageEnv(PushTEnv):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self,
            legacy=False,
            block_cog=None,
            damping=None,
            render_size=96,
            domain_filename=None,
            resize_scale=96, 
            pretrained=False):           # NEW CODE
        super().__init__(
            legacy=legacy,
            block_cog=block_cog,
            damping=damping,
            render_size=render_size,
            render_action=False,
            domain_filename=domain_filename)    # NEW CODE
        ws = self.window_size
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=1,
                shape=(3,render_size,render_size),
                dtype=np.float32
            ),
            'agent_pos': spaces.Box(
                low=0,
                high=ws,
                shape=(2,),
                dtype=np.float32
            )
        })
        self.render_cache = None
        self.resize_scale=resize_scale
        self.pretrained=pretrained

    def _get_obs(self):
        img = super()._render_frame(mode='rgb_array')

        if self.pretrained:
            transform = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(self.resize_scale),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            transform = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(self.resize_scale),
                v2.ToDtype(torch.float32, scale=True),
                # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        agent_pos = np.array(self.agent.position)

        # transform raw image
        img_obs = transform(img).numpy()
        # img_obs = np.moveaxis(img.astype(np.float32) / 255, -1, 0)
        obs = {
            'image': img_obs,
            'agent_pos': agent_pos,
            'goal_pose': (self.goal_pose[0], self.goal_pose[1], self.goal_pose[2] % (2*np.pi)),

        }

        # draw action
        if self.latest_action is not None:
            action = np.array(self.latest_action)
            coord = (action / 512 * 96).astype(np.int32)
            marker_size = int(8/96*self.render_size)
            thickness = int(1/96*self.render_size)
            cv2.drawMarker(img, coord,
                color=(255,0,0), markerType=cv2.MARKER_CROSS,
                markerSize=marker_size, thickness=thickness)
        self.render_cache = img

        return obs

    def render(self, mode):
        assert mode == 'rgb_array'

        if self.render_cache is None:
            self._get_obs()

        return self.render_cache

    def render_highres(self, size, draw_action_marker=True):
        """
        - High-resolution ground-truth render of the current state at `size`x`size`.
        - Decoupled from the agent observation (`resize_scale`) and the world-model conditioning
        (which stay at `render_size`), so this only affects display/eval videos.
        - When `draw_action_marker` is True, draws the red action marker at the correct scale for
        `size` (matching render()'s marker behavior); set it False for a clean, marker-free frame.
        """
        # render_action is False for PushTImageEnv, so the base render is marker-free.
        img = super()._render_frame(mode='rgb_array', out_size=size)
        if draw_action_marker and self.latest_action is not None:
            action = np.array(self.latest_action)
            coord = (action / 512 * size).astype(np.int32)
            marker_size = max(1, int(8/96*size))
            thickness = max(1, int(1/96*size))
            cv2.drawMarker(img, coord,
                color=(255,0,0), markerType=cv2.MARKER_CROSS,
                markerSize=marker_size, thickness=thickness)
        return img