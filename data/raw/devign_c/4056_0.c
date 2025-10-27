static int decode_slice(AVCodecContext *c, void *arg)

{

    FFV1Context *fs = *(void **)arg;

    FFV1Context *f  = fs->avctx->priv_data;

    int width, height, x, y, ret;

    const int ps = (av_pix_fmt_desc_get(c->pix_fmt)->flags & AV_PIX_FMT_FLAG_PLANAR)

                   ? (c->bits_per_raw_sample > 8) + 1

                   : 4;

    AVFrame *const p = f->cur;



    if (f->version > 2) {

        if (decode_slice_header(f, fs) < 0) {

            fs->slice_damaged = 1;

            return AVERROR_INVALIDDATA;

        }

    }

    if ((ret = ffv1_init_slice_state(f, fs)) < 0)

        return ret;

    if (f->cur->key_frame)

        ffv1_clear_slice_state(f, fs);

    width  = fs->slice_width;

    height = fs->slice_height;

    x      = fs->slice_x;

    y      = fs->slice_y;



    if (!fs->ac) {

        if (f->version == 3 && f->minor_version > 1 || f->version > 3)

            get_rac(&fs->c, (uint8_t[]) { 129 });

        fs->ac_byte_count = f->version > 2 || (!x && !y) ? fs->c.bytestream - fs->c.bytestream_start - 1 : 0;

        init_get_bits(&fs->gb, fs->c.bytestream_start + fs->ac_byte_count,

                      (fs->c.bytestream_end - fs->c.bytestream_start -

                       fs->ac_byte_count) * 8);

    }



    av_assert1(width && height);

    if (f->colorspace == 0) {

        const int chroma_width  = -((-width) >> f->chroma_h_shift);

        const int chroma_height = -((-height) >> f->chroma_v_shift);

        const int cx            = x >> f->chroma_h_shift;

        const int cy            = y >> f->chroma_v_shift;

        decode_plane(fs, p->data[0] + ps * x + y * p->linesize[0], width,

                     height, p->linesize[0],

                     0);



        if (f->chroma_planes) {

            decode_plane(fs, p->data[1] + ps * cx + cy * p->linesize[1],

                         chroma_width, chroma_height, p->linesize[1],

                         1);

            decode_plane(fs, p->data[2] + ps * cx + cy * p->linesize[2],

                         chroma_width, chroma_height, p->linesize[2],

                         1);

        }

        if (fs->transparency)

            decode_plane(fs, p->data[3] + ps * x + y * p->linesize[3], width,

                         height, p->linesize[3],

                         2);

    } else {

        uint8_t *planes[3] = { p->data[0] + ps * x + y * p->linesize[0],

                               p->data[1] + ps * x + y * p->linesize[1],

                               p->data[2] + ps * x + y * p->linesize[2] };

        decode_rgb_frame(fs, planes, width, height, p->linesize);

    }

    if (fs->ac && f->version > 2) {

        int v;

        get_rac(&fs->c, (uint8_t[]) { 129 });

        v = fs->c.bytestream_end - fs->c.bytestream - 2 - 5 * f->ec;

        if (v) {

            av_log(f->avctx, AV_LOG_ERROR, "bytestream end mismatching by %d\n",

                   v);

            fs->slice_damaged = 1;

        }

    }



    emms_c();



    return 0;

}
