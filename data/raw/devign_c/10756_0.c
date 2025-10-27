av_cold void ff_vp56_init(AVCodecContext *avctx, int flip, int has_alpha)

{

    VP56Context *s = avctx->priv_data;

    int i;



    s->avctx = avctx;

    avctx->pix_fmt = has_alpha ? PIX_FMT_YUVA420P : PIX_FMT_YUV420P;



    if (avctx->idct_algo == FF_IDCT_AUTO)

        avctx->idct_algo = FF_IDCT_VP3;

    ff_dsputil_init(&s->dsp, avctx);

    ff_vp56dsp_init(&s->vp56dsp, avctx->codec->id);

    ff_init_scantable(s->dsp.idct_permutation, &s->scantable,ff_zigzag_direct);



    for (i=0; i<4; i++)

        s->framep[i] = &s->frames[i];

    s->framep[VP56_FRAME_UNUSED] = s->framep[VP56_FRAME_GOLDEN];

    s->framep[VP56_FRAME_UNUSED2] = s->framep[VP56_FRAME_GOLDEN2];

    s->edge_emu_buffer_alloc = NULL;



    s->above_blocks = NULL;

    s->macroblocks = NULL;

    s->quantizer = -1;

    s->deblock_filtering = 1;



    s->filter = NULL;



    s->has_alpha = has_alpha;

    if (flip) {

        s->flip = -1;

        s->frbi = 2;

        s->srbi = 0;

    } else {

        s->flip = 1;

        s->frbi = 0;

        s->srbi = 2;

    }

}
