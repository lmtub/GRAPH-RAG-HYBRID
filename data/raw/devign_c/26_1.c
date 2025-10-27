static av_cold int iv_alloc_frames(Indeo3DecodeContext *s)

{

    int luma_width    = (s->width           + 3) & ~3,

        luma_height   = (s->height          + 3) & ~3,

        chroma_width  = ((luma_width  >> 2) + 3) & ~3,

        chroma_height = ((luma_height >> 2) + 3) & ~3,

        luma_pixels   = luma_width   * luma_height,

        chroma_pixels = chroma_width * chroma_height,

        i;

    unsigned int bufsize = luma_pixels * 2 + luma_width * 3 +

                          (chroma_pixels   + chroma_width) * 4;




    if(!(s->buf = av_malloc(bufsize)))

        return AVERROR(ENOMEM);

    s->iv_frame[0].y_w = s->iv_frame[1].y_w = luma_width;

    s->iv_frame[0].y_h = s->iv_frame[1].y_h = luma_height;

    s->iv_frame[0].uv_w = s->iv_frame[1].uv_w = chroma_width;

    s->iv_frame[0].uv_h = s->iv_frame[1].uv_h = chroma_height;



    s->iv_frame[0].Ybuf = s->buf + luma_width;

    i = luma_pixels + luma_width * 2;

    s->iv_frame[1].Ybuf = s->buf + i;

    i += (luma_pixels + luma_width);

    s->iv_frame[0].Ubuf = s->buf + i;

    i += (chroma_pixels + chroma_width);

    s->iv_frame[1].Ubuf = s->buf + i;

    i += (chroma_pixels + chroma_width);

    s->iv_frame[0].Vbuf = s->buf + i;

    i += (chroma_pixels + chroma_width);

    s->iv_frame[1].Vbuf = s->buf + i;



    for(i = 1; i <= luma_width; i++)

        s->iv_frame[0].Ybuf[-i] = s->iv_frame[1].Ybuf[-i] =

            s->iv_frame[0].Ubuf[-i] = 0x80;



    for(i = 1; i <= chroma_width; i++) {

        s->iv_frame[1].Ubuf[-i] = 0x80;

        s->iv_frame[0].Vbuf[-i] = 0x80;

        s->iv_frame[1].Vbuf[-i] = 0x80;

        s->iv_frame[1].Vbuf[chroma_pixels+i-1] = 0x80;

    }



    return 0;

}