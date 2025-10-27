static int64_t scene_sad16(FrameRateContext *s, const uint16_t *p1, int p1_linesize, const uint16_t* p2, int p2_linesize, int height)

{

    int64_t sad;

    int x, y;

    for (sad = y = 0; y < height; y += 8) {

        for (x = 0; x < p1_linesize; x += 8) {

            sad += sad_8x8_16(p1 + y * p1_linesize + x,

                              p1_linesize,

                              p2 + y * p2_linesize + x,

                              p2_linesize);

        }

    }

    return sad;

}
