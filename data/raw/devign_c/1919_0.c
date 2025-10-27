static void update_sono_yuv(AVFrame *sono, const ColorFloat *c, int idx)

{

    int x, fmt = sono->format, w = sono->width;

    uint8_t *lpy = sono->data[0] + idx * sono->linesize[0];

    uint8_t *lpu = sono->data[1] + idx * sono->linesize[1];

    uint8_t *lpv = sono->data[2] + idx * sono->linesize[2];



    for (x = 0; x < w; x += 2) {

        *lpy++ = c[x].yuv.y + 0.5f;

        *lpu++ = c[x].yuv.u + 0.5f;

        *lpv++ = c[x].yuv.v + 0.5f;

        *lpy++ = c[x+1].yuv.y + 0.5f;

        if (fmt == AV_PIX_FMT_YUV444P) {

            *lpu++ = c[x+1].yuv.u + 0.5f;

            *lpv++ = c[x+1].yuv.v + 0.5f;

        }

    }

}
