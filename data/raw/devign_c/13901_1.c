static void gif_fill_rect(AVFrame *picture, uint32_t color, int l, int t, int w, int h)

{

    const int linesize = picture->linesize[0] / sizeof(uint32_t);

    const uint32_t *py = (uint32_t *)picture->data[0] + t * linesize;

    const uint32_t *pr, *pb = py + (t + h) * linesize;

    uint32_t *px;



    for (; py < pb; py += linesize) {

        px = (uint32_t *)py + l;

        pr = px + w;



        for (; px < pr; px++)

            *px = color;

    }

}
