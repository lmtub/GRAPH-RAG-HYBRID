av_cold int ff_vp56_init_context(AVCodecContext *avctx, VP56Context *s,

                                  int flip, int has_alpha)

{

    int i;



    s->avctx = avctx;

    avctx->pix_fmt = has_alpha ? AV_PIX_FMT_YUVA420P : AV_PIX_FMT_YUV420P;

    if (avctx->skip_alpha) avctx->pix_fmt = AV_PIX_FMT_YUV420P;



    ff_h264chroma_init(&s->h264chroma, 8);

    ff_hpeldsp_init(&s->hdsp, avctx->flags);

    ff_videodsp_init(&s->vdsp, 8);

    ff_vp3dsp_init(&s->vp3dsp, avctx->flags);

    ff_vp56dsp_init(&s->vp56dsp, avctx->codec->id);

    for (i = 0; i < 64; i++) {

#define TRANSPOSE(x) (x >> 3) | ((x & 7) << 3)

        s->idct_scantable[i] = TRANSPOSE(ff_zigzag_direct[i]);

#undef TRANSPOSE

    }



    for (i = 0; i < FF_ARRAY_ELEMS(s->frames); i++) {

        s->frames[i] = av_frame_alloc();

        if (!s->frames[i]) {

            ff_vp56_free(avctx);

            return AVERROR(ENOMEM);

        }

    }

    s->edge_emu_buffer_alloc = NULL;



    s->above_blocks = NULL;

    s->macroblocks = NULL;

    s->quantizer = -1;

    s->deblock_filtering = 1;

    s->golden_frame = 0;



    s->filter = NULL;



    s->has_alpha = has_alpha;



    s->modelp = &s->model;



    if (flip) {

        s->flip = -1;

        s->frbi = 2;

        s->srbi = 0;

    } else {

        s->flip = 1;

        s->frbi = 0;

        s->srbi = 2;

    }



    return 0;

}
