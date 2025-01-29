from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
from lux.utils import direction_to
import numpy as np
import networkx as nx
from scipy.spatial import KDTree

@dataclass 
class GameState:
   relic_positions: List[np.ndarray]
   discovered_relics: Set[int]
   explore_tree: nx.Graph
   explored_nodes: Set[Tuple[int, int]]
   exploration_targets: Dict[int, Tuple[int, int]]
   energy_map: np.ndarray
   energy_threshold: int = 50  # Min energy before seeking recharge

class Agent:
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.team_id = int(player[-1])
        self.opp_team_id = 1 - self.team_id
        self.env_cfg = env_cfg
        np.random.seed(0)
        
        # Initialize with empty energy map
        self.state = GameState(
            [], set(), nx.Graph(), set(), {},
            np.zeros((env_cfg["map_width"], env_cfg["map_height"]))
        )

    def update_energy_map(self, obs_energy: np.ndarray, sensor_mask: np.ndarray):
        """Update known energy locations based on sensor observations"""
        visible = sensor_mask > 0
        self.state.energy_map[visible] = obs_energy[visible]

    def find_nearest_energy(self, unit_pos: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find closest high-energy tile"""
        energy_positions = np.where(self.state.energy_map > 5)
        if len(energy_positions[0]) == 0:
            return None
            
        positions = np.column_stack(energy_positions)
        tree = KDTree(positions)
        nearest_idx = tree.query([unit_pos])[1][0]
        return tuple(positions[nearest_idx])

    def get_unit_action(self, unit_id: int, unit_pos: np.ndarray, 
                       unit_energy: float, step: int) -> List[int]:
        """Determine action considering unit energy levels"""
        # Check if unit needs energy
        if unit_energy < self.state.energy_threshold:
            energy_pos = self.find_nearest_energy(unit_pos)
            if energy_pos:
                return [direction_to(unit_pos, energy_pos), 0, 0]
                
        # Original exploration/relic targeting logic
        if self.state.relic_positions:
            relic_pos = self.state.relic_positions[0]
            dist = abs(unit_pos[0] - relic_pos[0]) + abs(unit_pos[1] - relic_pos[1])

            if dist <= 4:
                return [np.random.randint(0, 5), 0, 0]
            return [direction_to(unit_pos, relic_pos), 0, 0]

        pos = tuple(map(int, unit_pos))
        self.state.explored_nodes.add(pos)

        if step % 20 == 0 or unit_id not in self.state.exploration_targets:
            target = self.get_next_exploration_target(unit_pos)
            self.state.exploration_targets[unit_id] = target

        return [direction_to(unit_pos, self.state.exploration_targets[unit_id]), 0, 0]

    def act(self, step: int, obs, remainingOverageTime: int = 60) -> np.ndarray:
        """Process observations and return actions for all units"""
        unit_mask = np.array(obs["units_mask"][self.team_id])
        units = obs["units"]
        unit_positions = np.array(units["position"][self.team_id])
        unit_energy = np.array(units["energy"][self.team_id])
        
        # Update known energy map
        self.update_energy_map(
            np.array(obs["map_features"]["energy"]),
            np.array(obs["sensor_mask"])
        )

        # Track relics
        relic_positions = np.array(obs["relic_nodes"])
        relic_mask = np.array(obs["relic_nodes_mask"])
        visible_relics = set(np.where(relic_mask)[0])
        self.update_relic_discoveries(visible_relics, relic_positions)

        # Get actions for each unit
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        for unit_id in np.where(unit_mask)[0]:
            actions[unit_id] = self.get_unit_action(
                unit_id, 
                unit_positions[unit_id],
                unit_energy[unit_id],
                step
            )

        return actions

    def get_next_exploration_target(self, unit_pos: np.ndarray) -> Tuple[int, int]:
        """Use RRT to find next unexplored area"""
        if len(self.state.explored_nodes) == 0:
            return (np.random.randint(0, self.env_cfg["map_width"]),
                    np.random.randint(0, self.env_cfg["map_height"]))

        # Build KD-tree of explored nodes for efficient nearest neighbor lookup
        explored = np.array(list(self.state.explored_nodes))
        tree = KDTree(explored)

        # Sample random points and find one furthest from explored areas
        samples = []
        for _ in range(10):
            point = (np.random.randint(0, self.env_cfg["map_width"]),
                     np.random.randint(0, self.env_cfg["map_height"]))
            dist = tree.query([point])[0][0]
            samples.append((dist, point))

        return max(samples, key=lambda x: x[0])[1]

    def update_relic_discoveries(self, visible_ids: Set[int], 
                                 relic_positions: np.ndarray) -> None:
        """Track newly discovered relic nodes"""
        for id in visible_ids:
            if id not in self.state.discovered_relics:
                self.state.discovered_relics.add(id)
                self.state.relic_positions.append(relic_positions[id])