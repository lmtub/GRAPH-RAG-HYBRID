static int idcin_decode_init(AVCodecContext *avctx)

{

    IdcinContext *s = avctx->priv_data;

    int i, j, histogram_index = 0;

    unsigned char *histograms;



    s->avctx = avctx;

    avctx->pix_fmt = PIX_FMT_PAL8;

    dsputil_init(&s->dsp, avctx);



    /* make sure the Huffman tables make it */

    if (s->avctx->extradata_size != HUFFMAN_TABLE_SIZE) {

        av_log(s->avctx, AV_LOG_ERROR, "  Id CIN video: expected extradata size of %d\n", HUFFMAN_TABLE_SIZE);

        return -1;

    }



    /* build the 256 Huffman decode trees */

    histograms = (unsigned char *)s->avctx->extradata;

    for (i = 0; i < 256; i++) {

        for(j = 0; j < HUF_TOKENS; j++)

            s->huff_nodes[i][j].count = histograms[histogram_index++];

        huff_build_tree(s, i);

    }



    s->frame.data[0] = NULL;



    return 0;

}
