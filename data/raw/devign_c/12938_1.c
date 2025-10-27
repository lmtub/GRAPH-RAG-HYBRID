static void calc_diffs(const DecimateContext *dm, struct qitem *q,

                       const AVFrame *f1, const AVFrame *f2)

{

    int64_t maxdiff = -1;

    int64_t *bdiffs = dm->bdiffs;

    int plane, i, j;



    memset(bdiffs, 0, dm->bdiffsize * sizeof(*bdiffs));



    for (plane = 0; plane < (dm->chroma ? 3 : 1); plane++) {

        int x, y, xl;

        const int linesize1 = f1->linesize[plane];

        const int linesize2 = f2->linesize[plane];

        const uint8_t *f1p = f1->data[plane];

        const uint8_t *f2p = f2->data[plane];

        int width    = plane ? FF_CEIL_RSHIFT(f1->width,  dm->hsub) : f1->width;

        int height   = plane ? FF_CEIL_RSHIFT(f1->height, dm->vsub) : f1->height;

        int hblockx  = dm->blockx / 2;

        int hblocky  = dm->blocky / 2;



        if (plane) {

            hblockx >>= dm->hsub;

            hblocky >>= dm->vsub;

        }



        for (y = 0; y < height; y++) {

            int ydest = y / hblocky;

            int xdest = 0;



#define CALC_DIFF(nbits) do {                               \

    for (x = 0; x < width; x += hblockx) {                  \

        int64_t acc = 0;                                    \

        int m = FFMIN(width, x + hblockx);                  \

        for (xl = x; xl < m; xl++)                          \

            acc += abs(((const uint##nbits##_t *)f1p)[xl] - \

                       ((const uint##nbits##_t *)f2p)[xl]); \

        bdiffs[ydest * dm->nxblocks + xdest] += acc;        \

        xdest++;                                            \

    }                                                       \

} while (0)

            if (dm->depth == 8) CALC_DIFF(8);

            else                CALC_DIFF(16);



            f1p += linesize1;

            f2p += linesize2;

        }

    }



    for (i = 0; i < dm->nyblocks - 1; i++) {

        for (j = 0; j < dm->nxblocks - 1; j++) {

            int64_t tmp = bdiffs[      i * dm->nxblocks + j    ]

                        + bdiffs[      i * dm->nxblocks + j + 1]

                        + bdiffs[(i + 1) * dm->nxblocks + j    ]

                        + bdiffs[(i + 1) * dm->nxblocks + j + 1];

            if (tmp > maxdiff)

                maxdiff = tmp;

        }

    }



    q->totdiff = 0;

    for (i = 0; i < dm->bdiffsize; i++)

        q->totdiff += bdiffs[i];

    q->maxbdiff = maxdiff;

}
