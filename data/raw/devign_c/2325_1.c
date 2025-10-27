static av_cold int init_bundles(BinkContext *c)

{

    int bw, bh, blocks;

    int i;



    bw = (c->avctx->width  + 7) >> 3;

    bh = (c->avctx->height + 7) >> 3;

    blocks = bw * bh;



    for (i = 0; i < BINKB_NB_SRC; i++) {

        c->bundle[i].data = av_malloc(blocks * 64);

        if (!c->bundle[i].data)

            return AVERROR(ENOMEM);

        c->bundle[i].data_end = c->bundle[i].data + blocks * 64;

    }



    return 0;

}
