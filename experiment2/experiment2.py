"""当代人工智能实验二：搜索算法

实现内容：
1. 普通网格 A*（返回路径）
2. 加权地形 A*（返回最小总代价和路径）
3. Minimax / Alpha-Beta（按左到右访问）
4. A* 距离矩阵 + 宝藏分配博弈（H=2 / H=4）

运行：python experiment2.py
仅使用 Python 标准库。
"""
from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from math import inf
from typing import Any, Optional
import json

Coord = tuple[int, int]

# 通用 A* 工具

def neighbors(pos: Coord, rows: int, cols: int):
    # 按右、下、左、上的顺序生成四个邻居，顺序只会影响同代价解的选择。
    r, c = pos
    for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc

def reconstruct_path(parent: dict[Coord, Coord], goal: Coord) -> list[Coord]:
    path = [goal]
    while path[-1] in parent:
        path.append(parent[path[-1]])
    path.reverse()
    return path

def astar_unweighted(
    grid: list[list[str]], start: Coord, goal: Coord, heuristic: str = "manhattan"
) -> tuple[list[Coord], dict[str, int]]:
    # 普通迷宫 A*。X 是障碍，其他格子均可通行，每步代价为 1。
    # heuristic 可取 manhattan 或 zero。expanded_nodes 统计实际展开邻居的节点数，不包含弹出后立即判定为目标的节点。
    rows, cols = len(grid), len(grid[0])
    if not (0 <= start[0] < rows and 0 <= start[1] < cols):
        return [], {"expanded_nodes": 0}
    if not (0 <= goal[0] < rows and 0 <= goal[1] < cols):
        return [], {"expanded_nodes": 0}
    if grid[start[0]][start[1]] == "X" or grid[goal[0]][goal[1]] == "X":
        return [], {"expanded_nodes": 0}

    def h(p: Coord) -> int:
        if heuristic == "zero":
            return 0
        return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

    serial = count()
    heap = [(h(start), h(start), next(serial), 0, start)]
    best_g = {start: 0}
    parent: dict[Coord, Coord] = {}
    expanded = 0

    while heap:
        _, _, _, g, current = heappop(heap)
        if g != best_g.get(current):
            continue
        if current == goal:
            return reconstruct_path(parent, goal), {"expanded_nodes": expanded}

        expanded += 1
        for nxt in neighbors(current, rows, cols):
            if grid[nxt[0]][nxt[1]] == "X":
                continue
            new_g = g + 1
            if new_g < best_g.get(nxt, inf):
                best_g[nxt] = new_g
                parent[nxt] = current
                hn = h(nxt)
                heappush(heap, (new_g + hn, hn, next(serial), new_g, nxt))

    return [], {"expanded_nodes": expanded}

def entry_cost(cell: Any) -> Optional[int]:
    # 进入一个格子的代价。X不能进入，E为0，A/B/T/S为1，数字按数字处理。
    if cell == "X":
        return None
    if cell == "E":
        return 0
    if cell in {"S", "A", "B", "T1", "T2", "T3", "T4"}:
        return 1
    try:
        return int(cell)
    except (TypeError, ValueError):
        raise ValueError(f"无法识别的地图格子：{cell!r}")

def astar_weighted(
    grid: list[list[Any]], start: Coord, goal: Coord, heuristic: str = "adjusted_manhattan"
) -> tuple[int, list[Coord], dict[str, int]]:
    # 加权 A*。移动代价等于进入相邻格子的代价，起点格不计费。
    # 若目标是E，进入E的代价为0。启发函数为 h = max(0, Manhattan距离 - 1) * 最小普通进入代价 + 目标格进入代价。该下界单独处理最后一步进入E免费的规则。
    rows, cols = len(grid), len(grid[0])
    if not (0 <= start[0] < rows and 0 <= start[1] < cols):
        return -1, [], {"expanded_nodes": 0}
    if not (0 <= goal[0] < rows and 0 <= goal[1] < cols):
        return -1, [], {"expanded_nodes": 0}
    if entry_cost(grid[start[0]][start[1]]) is None or entry_cost(grid[goal[0]][goal[1]]) is None:
        return -1, [], {"expanded_nodes": 0}

    goal_cost = entry_cost(grid[goal[0]][goal[1]])
    traversable_costs = [
        entry_cost(cell)
        for row in grid
        for cell in row
        if entry_cost(cell) is not None
    ]
    # 最小普通移动代价至少取 1；目标E的 0 不参与普通步代价的下界。
    positive_costs = [v for v in traversable_costs if v is not None and v > 0]
    min_step = min(positive_costs) if positive_costs else 1

    def h(p: Coord) -> int:
        if heuristic == "zero":
            return 0
        distance = abs(p[0] - goal[0]) + abs(p[1] - goal[1])
        if distance == 0:
            return 0
        if heuristic == "manhattan":
            # 用于对照
            return distance * min_step
        if heuristic != "adjusted_manhattan":
            raise ValueError("heuristic 必须是 zero/manhattan/adjusted_manhattan")
        return (distance - 1) * min_step + goal_cost

    serial = count()
    h0 = h(start)
    heap = [(h0, h0, next(serial), 0, start)]
    best_g = {start: 0}
    parent: dict[Coord, Coord] = {}
    expanded = 0

    while heap:
        _, _, _, g, current = heappop(heap)
        if g != best_g.get(current):
            continue
        if current == goal:
            return g, reconstruct_path(parent, goal), {"expanded_nodes": expanded}

        expanded += 1
        for nxt in neighbors(current, rows, cols):
            cost = entry_cost(grid[nxt[0]][nxt[1]])
            if cost is None:
                continue
            new_g = g + cost
            if new_g < best_g.get(nxt, inf):
                best_g[nxt] = new_g
                parent[nxt] = current
                hn = h(nxt)
                heappush(heap, (new_g + hn, hn, next(serial), new_g, nxt))

    return -1, [], {"expanded_nodes": expanded}


# 题目三：Minimax 和 Alpha-Beta

@dataclass
class TreeNode:
    value: Optional[int] = None
    children: Optional[list["TreeNode"]] = None
    is_maximizing_player: Optional[bool] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []

def tree_from_dict(data: dict[str, Any]) -> TreeNode:
    if "value" in data:
        return TreeNode(value=int(data["value"]), is_maximizing_player=data.get("is_maximizing_player"))
    children = [tree_from_dict(child) for child in data.get("children", [])]
    return TreeNode(children=children, is_maximizing_player=data.get("is_maximizing_player"))

def minimax_tree(node: TreeNode, maximizing: bool = True, stats: Optional[dict] = None) -> int:
    if node.is_maximizing_player is not None:
        maximizing = node.is_maximizing_player
    if stats is None:
        stats = {"visited_nodes": 0, "leaf_evaluations": 0, "cutoffs": 0}
    stats["visited_nodes"] += 1
    if not node.children:
        stats["leaf_evaluations"] += 1
        if node.value is None:
            raise ValueError("叶子节点必须提供 value")
        return node.value

    if maximizing:
        best = -inf
        for child in node.children:  # 必须按列表顺序，从左到右
            best = max(best, minimax_tree(child, False, stats))
        return int(best)
    else:
        best = inf
        for child in node.children:
            best = min(best, minimax_tree(child, True, stats))
        return int(best)

def alphabeta_tree(
    node: TreeNode,
    maximizing: bool = True,
    alpha: float = -inf,
    beta: float = inf,
    stats: Optional[dict] = None,
) -> int:
    if node.is_maximizing_player is not None:
        maximizing = node.is_maximizing_player
    if stats is None:
        stats = {"visited_nodes": 0, "leaf_evaluations": 0, "cutoffs": 0}
    stats["visited_nodes"] += 1
    if not node.children:
        stats["leaf_evaluations"] += 1
        if node.value is None:
            raise ValueError("叶子节点必须提供 value")
        return node.value

    if maximizing:
        value = -inf
        for i, child in enumerate(node.children):
            value = max(value, alphabeta_tree(child, False, alpha, beta, stats))
            alpha = max(alpha, value)
            if alpha >= beta and i < len(node.children) - 1:
                stats["cutoffs"] += 1
                stats["pruned_children"] = stats.get("pruned_children", 0) + (len(node.children) - i - 1)
                break
        return int(value)
    else:
        value = inf
        for i, child in enumerate(node.children):
            value = min(value, alphabeta_tree(child, True, alpha, beta, stats))
            beta = min(beta, value)
            if alpha >= beta and i < len(node.children) - 1:
                stats["cutoffs"] += 1
                stats["pruned_children"] = stats.get("pruned_children", 0) + (len(node.children) - i - 1)
                break
        return int(value)

def reverse_tree(node: TreeNode) -> TreeNode:
    # 复制一棵树，并把每个非叶子节点的孩子顺序反转（用于顺序对照实验）。
    if not node.children:
        return TreeNode(value=node.value, is_maximizing_player=node.is_maximizing_player)
    return TreeNode(children=[reverse_tree(child) for child in reversed(node.children)], is_maximizing_player=node.is_maximizing_player)

def reverse_children_at(node: TreeNode, target_path: tuple[int, ...], path: tuple[int, ...] = ()) -> TreeNode:
    # 仅反转指定路径节点的孩子顺序；路径(1,)表示根节点的右侧 MIN 子节点。
    if not node.children:
        return TreeNode(value=node.value, is_maximizing_player=node.is_maximizing_player)
    children = list(node.children)
    if path == target_path:
        children.reverse()
    return TreeNode(children=[reverse_children_at(child, target_path, path + (i,)) for i, child in enumerate(children)], is_maximizing_player=node.is_maximizing_player)


def sample_tree() -> TreeNode:
    # 叶子节点从左到右为 3,5 / 6,9 / 1,2。
    max1 = TreeNode(children=[TreeNode(value=3), TreeNode(value=5)], is_maximizing_player=True)
    max2_left = TreeNode(children=[TreeNode(value=6), TreeNode(value=9)], is_maximizing_player=True)
    max2_right = TreeNode(children=[TreeNode(value=6), TreeNode(value=9)], is_maximizing_player=True)
    max3 = TreeNode(children=[TreeNode(value=1), TreeNode(value=2)], is_maximizing_player=True)
    min1 = TreeNode(children=[max1, max2_left], is_maximizing_player=False)
    min2 = TreeNode(children=[max2_right, max3], is_maximizing_player=False)
    return TreeNode(children=[min1, min2], is_maximizing_player=True)


# 题目四：地图、距离矩阵和宝藏博弈

TREASURE_MAP = [
    ["A", 1, 1, "X", 2, 1, "T1"],
    [1, "X", 6, "X", 1, "X", 1],
    [1, 1, "T2", 1, 1, 1, 1],
    ["X", 1, "X", 8, "X", 1, "X"],
    [1, 1, 1, 1, "T3", 1, 1],
    [1, "X", 9, "X", 1, "X", 1],
    ["T4", 1, 1, "X", 1, 1, "B"],
]

POINTS = {
    "A": (0, 0),
    "B": (6, 6),
    "T1": (0, 6),
    "T2": (2, 2),
    "T3": (4, 4),
    "T4": (6, 0),
}
TREASURE_VALUES = {"T1": 9, "T2": 6, "T3": 7, "T4": 7}
POINT_ORDER = ["A", "B", "T1", "T2", "T3", "T4"]

def build_distance_matrix():
    matrix: dict[str, dict[str, int]] = {}
    path_details = {}
    for src_name in POINT_ORDER:
        matrix[src_name] = {}
        for dst_name in POINT_ORDER:
            dist, path, stats = astar_weighted(TREASURE_MAP, POINTS[src_name], POINTS[dst_name])
            matrix[src_name][dst_name] = dist
            path_details[(src_name, dst_name)] = {"distance": dist, "path": path, **stats}
    return matrix, path_details

@dataclass(frozen=True)
class GameState:
    pos_a: str
    pos_b: str
    remaining: tuple[str, ...]
    score_a: int
    score_b: int
    player: str

    @property
    def utility(self) -> int:
        return self.score_a - self.score_b

def move_order(state: GameState, distances: dict[str, dict[str, int]], mode: str) -> list[str]:
    moves = list(state.remaining)
    if mode == "natural":
        return moves
    if mode == "reverse":
        return list(reversed(moves))
    if mode == "heuristic":
        current_pos = state.pos_a if state.player == "A" else state.pos_b
        sign = 1 if state.player == "A" else -1
        # 估计当前着法对 U 的即时影响；MAX 优先看较大增量，MIN 优先看较小增量。
        moves.sort(
            key=lambda t: sign * (TREASURE_VALUES[t] - distances[current_pos][t]),
            reverse=True,
        )
        return moves
    raise ValueError("order_mode 必须是 natural/reverse/heuristic")

def apply_treasure(state: GameState, target: str, distances: dict[str, dict[str, int]]) -> GameState:
    if target not in state.remaining:
        raise ValueError(f"宝藏 {target} 已经被取走或不存在")
    if state.player == "A":
        d = distances[state.pos_a][target]
        gain = TREASURE_VALUES[target] - d
        return GameState(
            pos_a=target,
            pos_b=state.pos_b,
            remaining=tuple(t for t in state.remaining if t != target),
            score_a=state.score_a + gain,
            score_b=state.score_b,
            player="B",
        )
    d = distances[state.pos_b][target]
    gain = TREASURE_VALUES[target] - d
    return GameState(
        pos_a=state.pos_a,
        pos_b=target,
        remaining=tuple(t for t in state.remaining if t != target),
        score_a=state.score_a,
        score_b=state.score_b + gain,
        player="A",
    )

def minimax_game(
    distances: dict[str, dict[str, int]], depth: int, order_mode: str = "natural"
) -> tuple[dict[str, Any], dict[str, int]]:
    stats = {"visited_nodes": 1, "leaf_evaluations": 0, "cutoffs": 0, "pruned_children": 0}
    initial = GameState("A", "B", tuple(TREASURE_VALUES.keys()), 0, 0, "A")

    def search(state: GameState, left: int) -> int:
        stats["visited_nodes"] += 1
        if left == 0 or not state.remaining:
            stats["leaf_evaluations"] += 1
            return state.utility
        maximizing = state.player == "A"
        best = -inf if maximizing else inf
        for target in move_order(state, distances, order_mode):
            value = search(apply_treasure(state, target, distances), left - 1)
            best = max(best, value) if maximizing else min(best, value)
        return int(best)

    best_value = -inf
    best_target = None
    for target in move_order(initial, distances, order_mode):
        value = search(apply_treasure(initial, target, distances), depth - 1)
        if value > best_value:
            best_value, best_target = value, target
    return {"best_target": best_target, "best_value": int(best_value)}, stats

def alphabeta_game(
    distances: dict[str, dict[str, int]], depth: int, order_mode: str = "natural"
) -> tuple[dict[str, Any], dict[str, int]]:
    stats = {"visited_nodes": 1, "leaf_evaluations": 0, "cutoffs": 0, "pruned_children": 0}
    initial = GameState("A", "B", tuple(TREASURE_VALUES.keys()), 0, 0, "A")

    def search(state: GameState, left: int, alpha: float, beta: float) -> int:
        stats["visited_nodes"] += 1
        if left == 0 or not state.remaining:
            stats["leaf_evaluations"] += 1
            return state.utility
        maximizing = state.player == "A"
        best = -inf if maximizing else inf
        moves = move_order(state, distances, order_mode)
        for i, target in enumerate(moves):
            value = search(apply_treasure(state, target, distances), left - 1, alpha, beta)
            if maximizing:
                best = max(best, value)
                alpha = max(alpha, best)
            else:
                best = min(best, value)
                beta = min(beta, best)
            if alpha >= beta and i < len(moves) - 1:
                stats["cutoffs"] += 1
                stats["pruned_children"] += len(moves) - i - 1
                break
        return int(best)

    best_value = -inf
    best_target = None
    alpha, beta = -inf, inf
    root_moves = move_order(initial, distances, order_mode)
    for target in root_moves:
        value = search(apply_treasure(initial, target, distances), depth - 1, alpha, beta)
        if value > best_value:
            best_value, best_target = value, target
        alpha = max(alpha, best_value)
    return {"best_target": best_target, "best_value": int(best_value)}, stats

def run_all():
    print("题目一：迷宫寻路")
    maze = [
        ["S", ".", ".", "X", "."],
        [".", "X", ".", ".", "."],
        [".", "X", ".", "X", "."],
        [".", ".", ".", "X", "E"],
        ["X", "X", ".", ".", "."],
    ]
    path_m, stats_m = astar_unweighted(maze, (0, 0), (3, 4), "manhattan")
    path_0, stats_0 = astar_unweighted(maze, (0, 0), (3, 4), "zero")
    print("Manhattan path:", path_m)
    print("Manhattan stats:", stats_m)
    print("h=0 stats:", stats_0)

    print("\n题目二：加权地形")
    weighted = [
        ["S", 1, 1, 8, 1],
        [1, 8, 3, 1, 1],
        [1, 8, 1, 8, 1],
        [1, 1, 1, 8, 1],
        [8, 8, 1, 1, "E"],
    ]
    cost, path, stats = astar_weighted(weighted, (0, 0), (4, 4))
    print("minimum cost:", cost)
    print("path:", path)
    print("stats:", stats)
    cost_dij, _, stats_dij = astar_weighted(weighted, (0, 0), (4, 4), "zero")
    print("Dijkstra(h=0) cost/expanded:", cost_dij, stats_dij)

    print("\n题目三：Alpha-Beta")
    tree = sample_tree()
    s1 = {"visited_nodes": 0, "leaf_evaluations": 0, "cutoffs": 0}
    v1 = minimax_tree(tree, True, s1)
    s2 = {"visited_nodes": 0, "leaf_evaluations": 0, "cutoffs": 0}
    v2 = alphabeta_tree(tree, True, stats=s2)
    print("Minimax:", v1, s1)
    print("Alpha-Beta:", v2, s2)
    s3 = {"visited_nodes": 0, "leaf_evaluations": 0, "cutoffs": 0}
    v3 = alphabeta_tree(reverse_children_at(tree, (1,)), True, stats=s3)
    print("Alpha-Beta(仅反转右侧 MIN 的子节点，仅作顺序对照):", v3, s3)

    print("\n题目四：距离矩阵")
    distances, paths = build_distance_matrix()
    print("节点顺序:", POINT_ORDER)
    print(json.dumps([[distances[a][b] for b in POINT_ORDER] for a in POINT_ORDER], ensure_ascii=False))

    print("\n题目四：博弈结果")
    for depth in (2, 4):
        mm, mm_stats = minimax_game(distances, depth, "natural")
        ab, ab_stats = alphabeta_game(distances, depth, "natural")
        print(f"H={depth} Minimax:", mm, mm_stats)
        print(f"H={depth} Alpha-Beta:", ab, ab_stats)

    print("\n题目四：访问顺序对剪枝的影响")
    for mode in ("natural", "reverse", "heuristic"):
        result, stat = alphabeta_game(distances, 4, mode)
        print(mode, result, stat)


def run_custom_maze_tests():
    #自设迷宫与特殊情况测试，不改变讲义给定的四道题样例。
    print("\n自设拓展：小型迷宫与特殊情况测试")

    # 自设 5×5 迷宫：包含障碍物，验证算法能否在不同地图上找到最短路径。
    custom_maze = [
        ["S", ".", ".", "X", "."],
        ["X", "X", ".", "X", "."],
        [".", ".", ".", ".", "."],
        [".", "X", "X", "X", "."],
        [".", ".", ".", ".", "E"],
    ]
    start, goal = (0, 0), (4, 4)

    path_m, stats_m = astar_unweighted(
        custom_maze, start, goal, heuristic="manhattan"
    )
    path_0, stats_0 = astar_unweighted(
        custom_maze, start, goal, heuristic="zero"
    )

    print("自设迷宫：")
    for row in custom_maze:
        print(" ".join(str(cell) for cell in row))
    print("Manhattan path:", path_m)
    print("Manhattan steps:", len(path_m) - 1 if path_m else None)
    print("Manhattan stats:", stats_m)
    print("h=0 path:", path_0)
    print("h=0 steps:", len(path_0) - 1 if path_0 else None)
    print("h=0 stats:", stats_0)

    # 两种启发函数都应找到 8 步最短路径；比较搜索节点数时使用相同统计口径。
    assert path_m and path_0, "自设迷宫应当存在可行路径"
    assert path_m[0] == start and path_m[-1] == goal
    assert path_0[0] == start and path_0[-1] == goal
    assert len(path_m) - 1 == 8 and len(path_0) - 1 == 8

    # 特殊情况一：起点被障碍物完全封锁，应返回空路径。
    blocked_maze = [
        ["S", "X", "."],
        ["X", "X", "."],
        [".", ".", "E"],
    ]
    blocked_path, blocked_stats = astar_unweighted(
        blocked_maze, (0, 0), (2, 2), heuristic="manhattan"
    )
    print("\n不可达迷宫 path:", blocked_path)
    print("不可达迷宫 stats:", blocked_stats)
    assert blocked_path == [], "不可达迷宫应返回空列表"

    # 特殊情况二：起点和终点相同，路径只包含该坐标，不需要展开节点。
    same_path, same_stats = astar_unweighted(
        [["S"]], (0, 0), (0, 0), heuristic="manhattan"
    )
    print("起点等于终点 path:", same_path)
    print("起点等于终点 stats:", same_stats)
    assert same_path == [(0, 0)]

    print("[PASS] 自设迷宫、无解情况和起点等于终点三项测试全部通过。")


if __name__ == "__main__":
    run_all()
    run_custom_maze_tests()
