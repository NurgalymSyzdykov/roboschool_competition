import numpy as np
import heapq
import math
from typing import List, Tuple, Optional


class OccupancyGrid:
    def __init__(self, width_m: float = 20.0, height_m: float = 20.0, resolution: float = 0.1,
                 inflation_radius: float = 0.3):
        self.resolution = resolution
        self.cols = int(width_m / resolution)
        self.rows = int(height_m / resolution)

        self.grid = np.full((self.rows, self.cols), -1, dtype=np.int8)

        self.origin_x = width_m / 2.0
        self.origin_y = height_m / 2.0
        self.inflation_cells = int(inflation_radius / resolution)

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        col = int((x + self.origin_x) / self.resolution)
        row = int((y + self.origin_y) / self.resolution)
        col = max(0, min(col, self.cols - 1))
        row = max(0, min(row, self.rows - 1))
        return (row, col)

    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        x = (col * self.resolution) - self.origin_x + (self.resolution / 2.0)
        y = (row * self.resolution) - self.origin_y + (self.resolution / 2.0)
        return (x, y)

    def mark_free(self, x: float, y: float):
        row, col = self.world_to_grid(x, y)
        self.grid[row, col] = 0

    def mark_obstacle(self, x: float, y: float):
        row, col = self.world_to_grid(x, y)

        if self.grid[row, col] == 1:
            return

        self.grid[row, col] = 1

        if self.inflation_cells > 0:
            r_min = max(0, row - self.inflation_cells)
            r_max = min(self.rows - 1, row + self.inflation_cells)
            c_min = max(0, col - self.inflation_cells)
            c_max = min(self.cols - 1, col + self.inflation_cells)

            for r in range(r_min, r_max + 1):
                for c in range(c_min, c_max + 1):
                    if math.hypot(r - row, c - col) <= self.inflation_cells:
                        if self.grid[r, c] <= 0:
                            self.grid[r, c] = 2

    def is_free(self, row: int, col: int) -> bool:
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.grid[row, col] <= 0
        return False


class AStarPlanner:
    def __init__(self, grid_map: OccupancyGrid):
        self.map = grid_map
        self.neighbors = [
            (0, 1, 1.0), (1, 0, 1.0), (0, -1, 1.0), (-1, 0, 1.0),
            (1, 1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (-1, -1, 1.414)
        ]

    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.hypot(b[0] - a[0], b[1] - a[1])

    def plan(self, start_x: float, start_y: float, goal_x: float, goal_y: float) -> List[Tuple[float, float]]:
        start_idx = self.map.world_to_grid(start_x, start_y)
        goal_idx = self.map.world_to_grid(goal_x, goal_y)

        if not self.map.is_free(goal_idx[0], goal_idx[1]):
            return []

        frontier = []
        heapq.heappush(frontier, (0.0, start_idx))

        came_from = {start_idx: None}
        cost_so_far = {start_idx: 0.0}

        while frontier:
            current_f, current = heapq.heappop(frontier)

            if current == goal_idx:
                break

            for dx, dy, step_cost in self.neighbors:
                next_node = (current[0] + dx, current[1] + dy)

                if not self.map.is_free(next_node[0], next_node[1]):
                    continue

                new_cost = cost_so_far[current] + step_cost
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + self.heuristic(goal_idx, next_node)
                    heapq.heappush(frontier, (priority, next_node))
                    came_from[next_node] = current

        if goal_idx not in came_from:
            return []

        path = []
        current = goal_idx
        while current != start_idx:
            path.append(self.map.grid_to_world(current[0], current[1]))
            current = came_from[current]

        path.reverse()
        return path