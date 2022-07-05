from custom_types import *
import vtk
from ui import ui_utils
import constants
import abc


class SingleArrow(abc.ABC):

    def get_address(self):
        return self.mapper.GetAddressAsString('')

    @property
    def actor(self):
        return self.arrow[0]

    @property
    def mapper(self):
        return self.arrow[1]

    @property
    def base_points(self):
        return self.arrow[2]

    def turn_off(self):
        if self.on:
            self.render.RemoveActor(self.actor)
            self.on = False

    def turn_on(self):
        if not self.on:
            self.render.AddActor(self.actor)
            self.on = True

    def init_arrows(self, color):
        source = ui_utils.load_vtk_obj(self.arrow_path)
        init_points = source.GetPoints()
        actor, mapper = ui_utils.wrap_mesh(source, color)
        return actor, mapper, init_points

    @abc.abstractmethod
    def update_arrows_transform(self, gaussian):
        raise NotImplemented

    @abc.abstractmethod
    def get_transform(self):
        return ui_utils.Buttons.translate, self.direction

    @property
    @abc.abstractmethod
    def is_valid(self) -> bool:
        raise NotImplemented

    @abc.abstractmethod
    def get_p(self, direction: str) -> ARRAYS:
        raise NotImplemented

    @property
    @abc.abstractmethod
    def arrow_path(self) -> str:
        raise NotImplemented

    def __init__(self, render, direction: str, color: Tuple[float, float, float]):
        self.direction = direction
        self.p, self.mu = self.get_p(direction)
        self.render = render
        self.arrow = self.init_arrows(color)
        self.on = False


class TranslateArrow(SingleArrow):

    @property
    def is_valid(self) -> bool:
        return True

    def update_arrows_transform(self, gaussian):
        phi, mu, eigen, p = gaussian.gaussian.get_view_data()
        mu = mu + self.mu
        source = self.mapper.GetInput()
        source.SetPoints(self.base_points)
        transform = vtk.vtkTransform()
        mat = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                if i > 2:
                    mat.SetElement(i, j, 0)
                elif j > 2:
                    mat.SetElement(i, j, float(mu[i]))
                    # mat.SetElement(i, j, 0)
                else:
                    mat.SetElement(i, j, self.p[i, j])
                # mat_t[i, j] = mat.GetElement(i,j)
        mat.SetElement(3, 3, 1)
        transform.SetMatrix(mat)
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetInputData(source)
        transformFilter.SetTransform(transform)
        transformFilter.Update()
        self.mapper.SetInputConnection(transformFilter.GetOutputPort())

    def get_transform(self):
        return ui_utils.Buttons.translate, self.direction

    def get_p(self, direction: str) -> ARRAYS:
        mu = np.zeros(3)
        if direction == "right":
            p = np.eye(3)
        elif direction == "left":
            p = ui_utils.get_rotation_matrix(np.pi, 2)
        elif direction == "down":
            p = ui_utils.get_rotation_matrix(np.pi / 2, 1)
        elif direction == "up":
            p = ui_utils.get_rotation_matrix(3 * np.pi / 2, 1)
        elif direction == "a":
            p = ui_utils.get_rotation_matrix(np.pi / 2, 2)
            mu[1] = -.01
        elif direction == "z":
            p = ui_utils.get_rotation_matrix(3 * np.pi / 2, 2)
            mu[1] = .01
        else:
            raise ValueError
        return p * 0.005, mu

    @property
    def arrow_path(self) -> str:
        return f"{constants.DATA_ROOT}/ui_resources/translate_x2.obj"


class RotateArrow(SingleArrow):

    @property
    def axis(self):
        return {"left": 0, "right": 0, "up": 1, "down": 1, "a": 2, "z": 2}[self.direction]

    def update_arrows_transform(self, gaussian):
        phi, mu, eigen, p = gaussian.gaussian.get_raw_data()
        # self.is_valid_ = (eigen.argmax() == self.axis)
        if self.is_valid_:
            source = self.mapper.GetInput()
            source.SetPoints(self.base_points)
            transform = vtk.vtkTransform()
            mat = vtk.vtkMatrix4x4()
            # p = self.p
            p = np.matmul(p.transpose(), self.p) * .007
            # p = gaussian.gaussian.permute_p(p.transpose()) * 0.005
            for i in range(4):
                for j in range(4):
                    if i > 2:
                        mat.SetElement(i, j, 0)
                    elif j > 2:
                        mat.SetElement(i, j, float(mu[i]))
                        # mat.SetElement(i, j, 0)
                    else:
                        mat.SetElement(i, j, p[i, j])
                    # mat_t[i, j] = mat.GetElement(i,j)
            mat.SetElement(3, 3, 1)
            transform.SetMatrix(mat)
            transformFilter = vtk.vtkTransformPolyDataFilter()
            transformFilter.SetInputData(source)
            transformFilter.SetTransform(transform)
            transformFilter.Update()
            self.mapper.SetInputConnection(transformFilter.GetOutputPort())

    @property
    def is_valid(self) -> bool:
        return self.is_valid_

    def get_transform(self):
        return ui_utils.Buttons.rotate, self.direction

    @staticmethod
    def reflect(p, axis):
        p[axis] = - p[axis]
        return p

    def get_p(self, direction: str) -> ARRAYS:
        mu = np.zeros(3)
        if direction in ("right", "left"):
            p = np.matmul(ui_utils.get_rotation_matrix(np.pi / 2, 0),
                          ui_utils.get_rotation_matrix(np.pi / 2, 2))
            if direction == "left":
                p = self.reflect(p, 2)
        elif direction in ("down", "up"):
            p = np.matmul(ui_utils.get_rotation_matrix(np.pi / 2, 1),
                          ui_utils.get_rotation_matrix(np.pi / 2, 2))
            if direction == "up":
                p = self.reflect(p, 1)
        elif direction in ("a", "z"):
            p = np.eye(3)
            if direction == "a":
                p = self.reflect(p, 0)
        else:
            raise ValueError
        return p, mu

    @property
    def arrow_path(self) -> str:
        return f"{constants.DATA_ROOT}/ui_resources/rotate_x2.obj"

    def __init__(self, render, direction, color):
        super(RotateArrow, self).__init__(render, direction, color)
        self.is_valid_ = True


class ArrowManger:

    def turn_off(self):
        [arrow.turn_off() for arrow in self.arrows]
        self.on = False

    def turn_on(self):
        [arrow.turn_on() for arrow in self.arrows]
        self.on = True

    def update_arrows_transform(self, gaussian):
        self.turn_on()
        [arrow.update_arrows_transform(gaussian) for arrow in self.arrows]
        for arrow in self.arrows:
            if not arrow.is_valid:
                arrow.turn_off()

    def check_event(self, object_id):
        return object_id in self.addresses_dict

    def get_transform(self, object_id):
        return self.arrows[self.addresses_dict[object_id]].get_transform()

    @property
    def arrows(self) -> List[SingleArrow]:
        return self.arrows_[self.cur_arrows]

    @property
    def addresses_dict(self):
        return self.addresses_dict_[self.cur_arrows]

    def switch_arrows(self, arrow_type: ui_utils.Buttons):
        new_arrows = 0 if arrow_type is ui_utils.Buttons.translate else 1
        if new_arrows == self.cur_arrows:
            return False
        if self.on:
            self.turn_off()
            self.cur_arrows = new_arrows
            return True
        self.cur_arrows = new_arrows
        return False

    def __init__(self, render):
        self.on = False
        self.arrows_ = ([TranslateArrow(render, direction, color) for direction, color in
                                 zip(("left", "right", "up", "down", "a", "z"), ((1, 0, 0), (1, 0, 0),
                                                                                 (0, 1, 0), (0, 1, 0),
                                                                                 (0, 0, 1), (0, 0, 1)))],
                       [RotateArrow(render, direction, color) for direction, color in
                        zip(("left", "right", "up", "down", "a", "z"), ((1, 0, 0), (1, 0, 0),
                                                                        (0, 1, 0), (0, 1, 0),
                                                                        (0, 0, 1), (0, 0, 1)))])
        self.addresses_dict_ = [{arrows[i].get_address(): i for i in range(len(arrows))}
                                for arrows in self.arrows_]
        self.cur_arrows = 0
