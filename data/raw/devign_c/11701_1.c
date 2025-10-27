static int decode_hq_slice_row(AVCodecContext *avctx, void *arg, int jobnr, int threadnr)

{

    int i;

    DiracContext *s = avctx->priv_data;

    DiracSlice *slices = ((DiracSlice *)arg) + s->num_x*jobnr;

    for (i = 0; i < s->num_x; i++)

        decode_hq_slice(avctx, &slices[i]);

    return 0;

}
