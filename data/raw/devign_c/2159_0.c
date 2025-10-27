static void draw_bar(TestSourceContext *test, const uint8_t color[4],

                     int x, int y, int w, int h,

                     AVFrame *frame)

{

    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(frame->format);

    uint8_t *p, *p0;

    int plane;



    x = FFMIN(x, test->w - 1);

    y = FFMIN(y, test->h - 1);

    w = FFMIN(w, test->w - x);

    h = FFMIN(h, test->h - y);



    av_assert0(x + w <= test->w);

    av_assert0(y + h <= test->h);



    for (plane = 0; frame->data[plane]; plane++) {

        const int c = color[plane];

        const int linesize = frame->linesize[plane];

        int i, px, py, pw, ph;



        if (plane == 1 || plane == 2) {

            px = x >> desc->log2_chroma_w;

            pw = AV_CEIL_RSHIFT(w, desc->log2_chroma_w);

            py = y >> desc->log2_chroma_h;

            ph = AV_CEIL_RSHIFT(h, desc->log2_chroma_h);

        } else {

            px = x;

            pw = w;

            py = y;

            ph = h;

        }



        p0 = p = frame->data[plane] + py * linesize + px;

        memset(p, c, pw);

        p += linesize;

        for (i = 1; i < ph; i++, p += linesize)

            memcpy(p, p0, pw);

    }

}
