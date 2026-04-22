def _ring_threat_level(harmonies):
    """Return 0, 1, or 2 based on how close the player is to a harmony ring."""
    n_h = len(harmonies)
    if n_h < 3:
        return 0

    tile_degree = {}
    for p1, p2 in harmonies:
        tile_degree[p1] = tile_degree.get(p1, 0) + 1
        tile_degree[p2] = tile_degree.get(p2, 0) + 1
    ring_corners = sum(1 for d in tile_degree.values() if d >= 2)

    if n_h >= 4 and ring_corners >= 3:
        return 2
    if n_h >= 3 and ring_corners >= 2:
        return 1
    return 0


def _ring_completion_distance(harmonies):
    """Estimate minimum extra harmony pairs needed to close a ring."""
    if len(harmonies) < 2:
        return float('inf')

    adj = {}
    for p1, p2 in harmonies:
        adj.setdefault(p1, []).append(p2)
        adj.setdefault(p2, []).append(p1)

    def _dfs(node, visited):
        best = len(visited)
        for nb in adj.get(node, []):
            if nb not in visited:
                visited.add(nb)
                best = max(best, _dfs(nb, visited))
                visited.remove(nb)
        return best

    max_path = 0
    for start in adj:
        path_len = _dfs(start, {start})
        if path_len > max_path:
            max_path = path_len

    return max(0, 4 - max_path)
