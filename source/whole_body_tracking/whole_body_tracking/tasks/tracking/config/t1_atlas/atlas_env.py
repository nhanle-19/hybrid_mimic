from whole_body_tracking.tasks.tracking.config.t1_floating_model.floating_model_env import FloatingModelEnv
from whole_body_tracking.utils.atlas_controller import AtlasController


class AtlasEnv(FloatingModelEnv):
    def _create_controller(self, robot):
        return AtlasController(robot, self.cfg.hybrid_controller, self)

    def _reset_idx(self, env_ids):
        self.hybrid_controller.reset(env_ids)
        super()._reset_idx(env_ids)
