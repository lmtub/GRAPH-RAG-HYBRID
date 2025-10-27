static av_cold int pcm_dvd_decode_init(AVCodecContext *avctx)

{

    PCMDVDContext *s = avctx->priv_data;



    /* Invalid header to force parsing of the first header */

    s->last_header = -1;

    /* reserve space for 8 channels, 3 bytes/sample, 4 samples/block */

    if (!(s->extra_samples = av_malloc(8 * 3 * 4)))

        return AVERROR(ENOMEM);

    s->extra_sample_count = 0;



    return 0;

}
