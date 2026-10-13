"""Isaac Lab-Arena policy runner with the UR5e + Robotiq 3F embodiments registered.

Same arguments as isaaclab_arena/evaluation/policy_runner.py; additionally accepts
--embodiment ur5e_3f_ik | ur5e_3f_ik_abs | ur5e_3f_joint_pos in any Arena environment.

    cd ~/IsaacLab-Arena && source .venv/bin/activate
    python ~/ur5eredodo/arena/run_arena.py --viz kit --policy_type zero_action --num_steps 300 \\
        pick_and_place_maple_table --embodiment ur5e_3f_ik
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import isaaclab_arena.evaluation.policy_runner as policy_runner  # noqa: E402

_get_arena_builder_from_cli = policy_runner.get_arena_builder_from_cli


def _get_arena_builder_with_ur5e(*args, **kwargs):
    # Isaac Lab modules exist only once the SimulationApp is running, so register here rather than at import.
    import ur5e_arena.embodiment  # noqa: F401

    return _get_arena_builder_from_cli(*args, **kwargs)


policy_runner.get_arena_builder_from_cli = _get_arena_builder_with_ur5e

if __name__ == "__main__":
    policy_runner.main()
