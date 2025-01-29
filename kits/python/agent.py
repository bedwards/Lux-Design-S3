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
   
class Agent:
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.team_id = int(player[-1])
        self.opp_team_id = 1 - self.team_id
        self.env_cfg = env_cfg
        np.random.seed(0)

        # Initialize RRT exploration graph
        self.state = GameState([], set(), nx.Graph(), set(), {})
       
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

    def get_unit_action(self, unit_id: int, unit_pos: np.ndarray, step: int) -> List[int]:
        """Determine action using RRT exploration or relic targeting"""
        if self.state.relic_positions:
            relic_pos = self.state.relic_positions[0]
            dist = abs(unit_pos[0] - relic_pos[0]) + abs(unit_pos[1] - relic_pos[1])

            if dist <= 4:
                return [np.random.randint(0, 5), 0, 0]
            return [direction_to(unit_pos, relic_pos), 0, 0]

        # Update exploration target every 20 steps
        pos = tuple(map(int, unit_pos))
        self.state.explored_nodes.add(pos)

        if step % 20 == 0 or unit_id not in self.state.exploration_targets:
            target = self.get_next_exploration_target(unit_pos)
            self.state.exploration_targets[unit_id] = target

        return [direction_to(unit_pos, self.state.exploration_targets[unit_id]), 0, 0]

    def update_relic_discoveries(self, visible_ids: Set[int], 
                                 relic_positions: np.ndarray) -> None:
        """Track newly discovered relic nodes"""
        for id in visible_ids:
            if id not in self.state.discovered_relics:
                self.state.discovered_relics.add(id)
                self.state.relic_positions.append(relic_positions[id])

    def act(self, step: int, obs, remainingOverageTime: int = 60) -> np.ndarray:
        """Process observations and return actions for all units"""
        unit_mask = np.array(obs["units_mask"][self.team_id])
        unit_positions = np.array(obs["units"]["position"][self.team_id])
        relic_positions = np.array(obs["relic_nodes"])
        relic_mask = np.array(obs["relic_nodes_mask"])

        visible_relics = set(np.where(relic_mask)[0])
        self.update_relic_discoveries(visible_relics, relic_positions)

        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        for unit_id in np.where(unit_mask)[0]:
            actions[unit_id] = self.get_unit_action(
                unit_id, unit_positions[unit_id], step)

        return actions