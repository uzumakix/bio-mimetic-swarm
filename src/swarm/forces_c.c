/*
 * forces_c.c -- Lennard-Jones force computation for swarm agents
 *
 * Build:
 *   gcc -O2 -shared -fPIC -o forces_c.so forces_c.c    (Linux/macOS)
 *   gcc -O2 -shared -o forces_c.dll forces_c.c          (Windows)
 */

#include <math.h>

#define MIN_DIST 0.1

/*
 * Compute pairwise Lennard-Jones forces for N agents in 2D.
 *
 * pos:       flat input array [x0, y0, x1, y1, ...], length 2*N
 * forces:    flat output array [fx0, fy0, ...], length 2*N (zeroed internally)
 * n:         number of agents
 * epsilon:   LJ well depth
 * sigma:     LJ zero-crossing distance
 * max_force: clamp per-pair force magnitude to this value
 *
 * Uses 12-6 LJ: F = 24*eps * (2*(sig/r)^13 - (sig/r)^7) / r
 * Newton's third law applied -- each pair computed once.
 */
#ifdef _WIN32
__declspec(dllexport)
#endif
void compute_lj_forces(
    const double *pos, double *forces, int n,
    double epsilon, double sigma, double max_force
) {
    int i, j;
    double dx, dy, r2, r, inv_r, sr, sr6, sr12;
    double fmag, fx, fy, scale;

    /* zero output */
    for (i = 0; i < 2 * n; i++) {
        forces[i] = 0.0;
    }

    for (i = 0; i < n; i++) {
        double xi = pos[2 * i];
        double yi = pos[2 * i + 1];

        for (j = i + 1; j < n; j++) {
            dx = pos[2 * j] - xi;
            dy = pos[2 * j + 1] - yi;

            r2 = dx * dx + dy * dy;
            r = sqrt(r2);

            /* floor to avoid singularity */
            if (r < MIN_DIST) {
                r = MIN_DIST;
            }

            inv_r = 1.0 / r;
            sr = sigma * inv_r;
            sr6 = sr * sr * sr * sr * sr * sr;
            sr12 = sr6 * sr6;

            /* scalar force magnitude along the pair axis */
            fmag = 24.0 * epsilon * (2.0 * sr12 - sr6) * inv_r;

            /* clamp */
            if (fmag > max_force) {
                fmag = max_force;
            } else if (fmag < -max_force) {
                fmag = -max_force;
            }

            /* direction: unit vector from i to j */
            fx = fmag * dx * inv_r;
            fy = fmag * dy * inv_r;

            /* Newton's third law */
            forces[2 * i]     += fx;
            forces[2 * i + 1] += fy;
            forces[2 * j]     -= fx;
            forces[2 * j + 1] -= fy;
        }
    }
}
