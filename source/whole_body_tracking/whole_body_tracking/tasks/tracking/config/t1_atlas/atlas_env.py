from whole_body_tracking.tasks.tracking.config.t1_floating_model.floating_model_env import FloatingModelEnv


class AtlasEnv(FloatingModelEnv):
    def _create_controller(self, robot):
        backend = self.cfg.hybrid_controller.backend
        if backend == 'batched':
            from whole_body_tracking.utils.atlas_batched_controller import AtlasBatchedController
            return AtlasBatchedController(robot, self.cfg.hybrid_controller, self)
        if backend == 'osqp':
            from whole_body_tracking.utils.atlas_controller import AtlasController
            return AtlasController(robot, self.cfg.hybrid_controller, self)
        raise ValueError(f'Unknown Atlas backend: {backend}')

    def _reset_idx(self, env_ids):
        self.hybrid_controller.reset(env_ids)
        super()._reset_idx(env_ids)
