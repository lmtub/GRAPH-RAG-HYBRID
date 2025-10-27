static int64_t scene_sad8(FrameRateContext *s, uint8_t *p1, int p1_linesize, uint8_t* p2, int p2_linesize, int height)

{

    int64_t sad;

    int x, y;

    for (sad = y = 0; y < height; y += 8) {

        for (x = 0; x < p1_linesize; x += 8) {

            sad += s->sad(p1 + y * p1_linesize + x,

                          p1_linesize,

                          p2 + y * p2_linesize + x,

                          p2_linesize);

        }

    }

    emms_c();

    return sad;

}
