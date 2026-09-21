import numpy as np


def response_points(load, displacement):
    load = np.asarray(load, dtype=float)
    displacement = np.asarray(displacement, dtype=float)

    mask = np.isfinite(load) & np.isfinite(displacement)
    load = load[mask]
    displacement = displacement[mask]

    if len(load) == 0:
        return None

    peak_index = int(np.argmax(load))
    peak_load = float(load[peak_index])
    peak_displacement = float(displacement[peak_index])

    threshold = (2.0 / 3.0) * peak_load
    crossings = np.where(load >= threshold)[0]

    yield_displacement = np.nan
    yield_load = np.nan

    if len(crossings):
        i = int(crossings[0])
        x1 = float(displacement[i])
        y1 = float(load[i])

        if x1 != 0 and y1 != 0:
            slope = y1 / x1
            yield_displacement = peak_load / slope

            order = np.argsort(displacement)
            x_sorted = displacement[order]
            y_sorted = load[order]

            if x_sorted.min() <= yield_displacement <= x_sorted.max():
                yield_load = float(np.interp(yield_displacement, x_sorted, y_sorted))

    final_displacement = float(displacement[-1])
    final_load = float(load[-1])

    ductility = (
        final_displacement / yield_displacement
        if np.isfinite(yield_displacement) and yield_displacement != 0
        else np.nan
    )

    stiffness = (
        yield_load / yield_displacement
        if np.isfinite(yield_load)
        and np.isfinite(yield_displacement)
        and yield_displacement != 0
        else np.nan
    )

    return {
        "peak_load": peak_load,
        "peak_displacement": peak_displacement,
        "yield_load": yield_load,
        "yield_displacement": yield_displacement,
        "final_load": final_load,
        "final_displacement": final_displacement,
        "ductility": ductility,
        "initial_stiffness": stiffness,
    }
