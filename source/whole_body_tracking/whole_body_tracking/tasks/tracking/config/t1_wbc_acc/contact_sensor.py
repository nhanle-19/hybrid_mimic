"""Ground-filtered wrench reporting compatible with the installed Isaac Lab."""
from isaaclab.sensors import ContactSensor


class WBCACCContactSensor(ContactSensor):
    def _initialize_impl(self):
        super()._initialize_impl()
        # Older Isaac Lab allocates zero detailed-contact capacity by default.
        # Each WBCACC sensor resolves exactly one foot per environment.
        self._contact_physx_view = self._physics_sim_view.create_rigid_contact_view(
            self.cfg.prim_path.replace('.*', '*'),
            filter_patterns=[p.replace('.*', '*') for p in self.cfg.filter_prim_paths_expr],
            max_contact_data_count=32*self._num_envs)
