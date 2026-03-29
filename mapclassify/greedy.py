"""
greedy - Greedy (topological) coloring for GeoPandas

Copyright (C) 2019 Martin Fleischmann, 2017 Nyall Dawson
"""

import operator

__all__ = ["greedy"]


def _balanced(features, sw, balance="count", min_colors=4):
    """
    Strategy to color features in a way which is visually balanced.
    """
    feature_colors = {}
    color_pool = set(range(min_colors))
    neighbour_count = sw.cardinalities

    sorted_by_count = sorted(
        neighbour_count.items(), key=operator.itemgetter(1), reverse=True
    )

    color_counts = {}
    color_areas = {}
    for c in color_pool:
        color_counts[c] = 0
        color_areas[c] = 0

    if balance == "centroid":
        features = features.copy()
        features.geometry = features.geometry.centroid
        balance = "distance"

    for feature_id, _ in sorted_by_count:
        adjacent_colors = set()
        for neighbour in sw.neighbors[feature_id]:
            if neighbour in feature_colors:
                adjacent_colors.add(feature_colors[neighbour])

        available_colors = color_pool.difference(adjacent_colors)

        feature_color = -1
        if len(available_colors) == 0:
            min_colors += 1
            return _balanced(features, sw, balance, min_colors)
        else:
            if balance == "count":
                counts = [
                    (c, v) for c, v in color_counts.items() if c in available_colors
                ]
                feature_color = sorted(counts, key=operator.itemgetter(1))[0][0]
                color_counts[feature_color] += 1

            elif balance == "area":
                areas = [
                    (c, v) for c, v in color_areas.items() if c in available_colors
                ]
                feature_color = sorted(areas, key=operator.itemgetter(1))[0][0]
                color_areas[feature_color] += features.loc[feature_id].geometry.area

            elif balance == "distance":
                min_distances = {c: float("inf") for c in available_colors}
                this_feature = features.loc[feature_id].geometry

                other_features = {
                    f_id: c
                    for (f_id, c) in feature_colors.items()
                    if c in available_colors
                }

                distances = features.loc[other_features.keys()].distance(this_feature)

                for other_feature_id, c in other_features.items():
                    distance = distances.loc[other_feature_id]
                    if distance < min_distances[c]:
                        min_distances[c] = distance

                feature_color = sorted(
                    min_distances, key=min_distances.__getitem__, reverse=True
                )[0]

        feature_colors[feature_id] = feature_color

    return feature_colors


def greedy(
    gdf,
    strategy: str = "balanced",
    balance: str = "count",
    min_colors: int = 4,
    sw="queen",
    min_distance: float | None = None,
    silence_warnings: bool = True,
    interchange: bool = False,
):
    """
    Color GeoDataFrame using greedy (topological) colouring.
    """

    if strategy != "balanced":
        try:
            import networkx as nx

            strategies = nx.algorithms.coloring.greedy_coloring.STRATEGIES.keys()
        except ImportError:
            raise ImportError("The 'networkx' package is required.") from None
    else:
        strategies = []

    try:
        import pandas as pd
    except ImportError:
        raise ImportError("The 'pandas' package is required.") from None

    try:
        from libpysal.weights import Queen, Rook, W, fuzzy_contiguity
    except ImportError:
        raise ImportError("The 'libpysal' package is required.") from None

    if min_distance is not None:
        sw = fuzzy_contiguity(
            gdf,
            tolerance=0.0,
            buffering=True,
            buffer=min_distance / 2.0,
            silence_warnings=silence_warnings,
        )

    if not isinstance(sw, W):
        if sw == "queen":
            sw = Queen.from_dataframe(
                gdf, silence_warnings=silence_warnings, use_index=False
            )
        elif sw == "rook":
            sw = Rook.from_dataframe(
                gdf, silence_warnings=silence_warnings, use_index=False
            )

    if strategy == "balanced":
        color = pd.Series(_balanced(gdf, sw, balance=balance, min_colors=min_colors))

    elif strategy in strategies:
        color = nx.greedy_color(
            sw.to_networkx(), strategy=strategy, interchange=interchange
        )

    else:
        raise ValueError(
            f"'{strategy}' is not a valid strategy. "
            "Use a supported NetworkX greedy coloring strategy or 'balanced'."
        )

    color = pd.Series(color).sort_index()
    color.index = gdf.index
    return color
