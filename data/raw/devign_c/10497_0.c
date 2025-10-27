static void randomize_loopfilter_buffers(int bidx, int lineoff, int str,

                                         int bit_depth, int dir,

                                         int* E, int* F, int* H, int* I,

                                         uint8_t *buf0, uint8_t *buf1)

{

    uint32_t mask = (1 << bit_depth) - 1;

    int off = dir ? lineoff : lineoff * 16;

    int istride = dir ? 1 : 16;

    int jstride = dir ? str : 1;

    int i, j;

    for (i = 0; i < 2; i++) /* flat16 */ {

        int idx = off + i * istride, p0, q0;

        setpx(idx,  0, q0 = rnd() & mask);

        setsx(idx, -1, p0 = q0, E[bidx] >> 2);

        for (j = 1; j < 8; j++) {

            setsx(idx, -1 - j, p0, F[bidx]);

            setsx(idx, j, q0, F[bidx]);

        }

    }

    for (i = 2; i < 4; i++) /* flat8 */ {

        int idx = off + i * istride, p0, q0;

        setpx(idx,  0, q0 = rnd() & mask);

        setsx(idx, -1, p0 = q0, E[bidx] >> 2);

        for (j = 1; j < 4; j++) {

            setsx(idx, -1 - j, p0, F[bidx]);

            setsx(idx, j, q0, F[bidx]);

        }

        for (j = 4; j < 8; j++) {

            setpx(idx, -1 - j, rnd() & mask);

            setpx(idx, j, rnd() & mask);

        }

    }

    for (i = 4; i < 6; i++) /* regular */ {

        int idx = off + i * istride, p2, p1, p0, q0, q1, q2;

        setpx(idx,  0, q0 = rnd() & mask);

        setsx(idx,  1, q1 = q0, I[bidx]);

        setsx(idx,  2, q2 = q1, I[bidx]);

        setsx(idx,  3, q2,      I[bidx]);

        setsx(idx, -1, p0 = q0, E[bidx] >> 2);

        setsx(idx, -2, p1 = p0, I[bidx]);

        setsx(idx, -3, p2 = p1, I[bidx]);

        setsx(idx, -4, p2,      I[bidx]);

        for (j = 4; j < 8; j++) {

            setpx(idx, -1 - j, rnd() & mask);

            setpx(idx, j, rnd() & mask);

        }

    }

    for (i = 6; i < 8; i++) /* off */ {

        int idx = off + i * istride;

        for (j = 0; j < 8; j++) {

            setpx(idx, -1 - j, rnd() & mask);

            setpx(idx, j, rnd() & mask);

        }

    }

}
