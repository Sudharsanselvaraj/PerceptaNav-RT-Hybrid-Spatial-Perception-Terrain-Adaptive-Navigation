"""
Path Planner
A*-based grid path planning using traversability map from TerrainClassifier.
Outputs waypoints for rover to follow toward selected target.
"""

import heapq
import numpy as np
import cv2
import logging
from typing import List, Tuple, Optional

log = logging.getLogger("PathPlanner")

# Grid cell cost = 1 / traversability  (lower = easier to traverse)
INF = float("inf")


def _heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(cost_grid: np.ndarray,
          start: Tuple[int, int],
          goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    """
    A* on a cost grid.
    cost_grid[r][c] = traversal cost (INF = blocked).
    Returns list of (row, col) or None if no path.
    """
    rows, cols = cost_grid.shape
    open_heap  = []
    heapq.heappush(open_heap, (0 + _heuristic(start, goal), 0, start))
    came_from  = {}
    g_score    = {start: 0.0}

    while open_heap:
        _, g, current = heapq.heappop(open_heap)

        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        r, c = current
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            cell_cost = cost_grid[nr, nc]
            if cell_cost >= INF:
                continue
            step = 1.414 if dr != 0 and dc != 0 else 1.0
            new_g = g + cell_cost * step
            if new_g < g_score.get((nr, nc), INF):
                g_score[(nr, nc)] = new_g
                came_from[(nr, nc)] = current
                f = new_g + _heuristic((nr, nc), goal)
                heapq.heappush(open_heap, (f, new_g, (nr, nc)))

    return None


class PathPlanner:
    """
    Builds a cost grid from terrain traversability and finds
    the optimal path from rover position to the selected target.
    """

    def __init__(self, grid_rows: int = 4, grid_cols: int = 6):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self._last_path: Optional[List[Tuple[int, int]]] = None

    def build_cost_grid(self, terrain_grid) -> np.ndarray:
        cost = np.zeros((self.grid_rows, self.grid_cols))
        for r, row in enumerate(terrain_grid):
            for c, patch in enumerate(row):
                t = patch.traversability
                cost[r, c] = (1.0 / max(t, 0.01)) if t > 0.05 else INF
        return cost

    def plan(self, terrain_grid,
             target_col: int) -> Optional[List[Tuple[int, int]]]:
        """
        Plans path from bottom-center of frame (rover position)
        to target column at top of frame.
        """
        cost_grid = self.build_cost_grid(terrain_grid)
        start     = (self.grid_rows - 1, self.grid_cols // 2)
        goal      = (0, int(np.clip(target_col, 0, self.grid_cols - 1)))

        path = astar(cost_grid, start, goal)
        self._last_path = path

        if path:
            log.debug(f"Path found: {len(path)} steps → col {target_col}")
        else:
            log.warning(f"No feasible path to col {target_col}")

        return path

    def draw_path(self, frame: np.ndarray,
                  terrain_grid,
                  path: Optional[List[Tuple[int, int]]]) -> np.ndarray:
        if not path:
            return frame

        h, w = frame.shape[:2]
        ph   = h // self.grid_rows
        pw   = w // self.grid_cols

        pts = []
        for r, c in path:
            px = c * pw + pw // 2
            py = r * ph + ph // 2
            pts.append((px, py))

        for i in range(len(pts) - 1):
            cv2.line(frame, pts[i], pts[i + 1], (0, 255, 255), 2)

        for pt in pts:
            cv2.circle(frame, pt, 4, (0, 200, 255), -1)

        # Arrow at start
        if len(pts) >= 2:
            cv2.arrowedLine(frame, pts[-1], pts[-2], (0, 255, 200), 2,
                            tipLength=0.4)

        return frame

    def steering_command(self, path: Optional[List[Tuple[int, int]]]) -> str:
        """
        Translates first path step into a steering command.
        Returns: 'FORWARD' | 'LEFT' | 'RIGHT' | 'STOP'
        """
        if not path or len(path) < 2:
            return "STOP"
        _, c_now  = path[0]
        _, c_next = path[1]
        if c_next < c_now:
            return "LEFT"
        elif c_next > c_now:
            return "RIGHT"
        return "FORWARD"
