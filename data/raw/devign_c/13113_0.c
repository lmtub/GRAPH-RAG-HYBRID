static int encode_slice(AVCodecContext *c, void *arg)

{

    FFV1Context *fs  = *(void **)arg;

    FFV1Context *f   = fs->avctx->priv_data;

    int width        = fs->slice_width;

    int height       = fs->slice_height;

    int x            = fs->slice_x;

    int y            = fs->slice_y;

    const AVFrame *const p = f->frame;

    const int ps     = (av_pix_fmt_desc_get(c->pix_fmt)->flags & AV_PIX_FMT_FLAG_PLANAR)

                       ? (f->bits_per_raw_sample > 8) + 1

                       : 4;



    if (f->key_frame)

        ffv1_clear_slice_state(f, fs);

    if (f->version > 2) {

        encode_slice_header(f, fs);

    }

    if (!fs->ac) {

        if (f->version > 2)

            put_rac(&fs->c, (uint8_t[]) { 129 }, 0);

        fs->ac_byte_count = f->version > 2 || (!x && !y) ? ff_rac_terminate( &fs->c) : 0;

        init_put_bits(&fs->pb, fs->c.bytestream_start + fs->ac_byte_count,

                      fs->c.bytestream_end - fs->c.bytestream_start - fs->ac_byte_count);

    }



    if (f->colorspace == 0) {

        const int chroma_width  = -((-width) >> f->chroma_h_shift);

        const int chroma_height = -((-height) >> f->chroma_v_shift);

        const int cx            = x >> f->chroma_h_shift;

        const int cy            = y >> f->chroma_v_shift;



        encode_plane(fs, p->data[0] + ps * x + y * p->linesize[0],

                     width, height, p->linesize[0], 0);



        if (f->chroma_planes) {

            encode_plane(fs, p->data[1] + ps * cx + cy * p->linesize[1],

                         chroma_width, chroma_height, p->linesize[1], 1);

            encode_plane(fs, p->data[2] + ps * cx + cy * p->linesize[2],

                         chroma_width, chroma_height, p->linesize[2], 1);

        }

        if (fs->transparency)

            encode_plane(fs, p->data[3] + ps * x + y * p->linesize[3], width,

                         height, p->linesize[3], 2);

    } else {

        const uint8_t *planes[3] = { p->data[0] + ps * x + y * p->linesize[0],

                                     p->data[1] + ps * x + y * p->linesize[1],

                                     p->data[2] + ps * x + y * p->linesize[2] };

        encode_rgb_frame(fs, planes, width, height, p->linesize);

    }

    emms_c();



    return 0;

}
