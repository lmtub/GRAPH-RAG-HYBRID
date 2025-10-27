static int decode_init_mp3on4(AVCodecContext * avctx)

{

    MP3On4DecodeContext *s = avctx->priv_data;

    int i;



    if ((avctx->extradata_size < 2) || (avctx->extradata == NULL)) {

        av_log(avctx, AV_LOG_ERROR, "Codec extradata missing or too short.\n");

        return -1;

    }



    s->chan_cfg = (((unsigned char *)avctx->extradata)[1] >> 3) & 0x0f;

    s->frames = mp3Frames[s->chan_cfg];

    if(!s->frames) {

        av_log(avctx, AV_LOG_ERROR, "Invalid channel config number.\n");

        return -1;

    }

    avctx->channels = mp3Channels[s->chan_cfg];



    /* Init the first mp3 decoder in standard way, so that all tables get builded

     * We replace avctx->priv_data with the context of the first decoder so that

     * decode_init() does not have to be changed.

     * Other decoders will be inited here copying data from the first context

     */

    // Allocate zeroed memory for the first decoder context

    s->mp3decctx[0] = av_mallocz(sizeof(MPADecodeContext));

    // Put decoder context in place to make init_decode() happy

    avctx->priv_data = s->mp3decctx[0];

    decode_init(avctx);

    // Restore mp3on4 context pointer

    avctx->priv_data = s;

    s->mp3decctx[0]->adu_mode = 1; // Set adu mode



    /* Create a separate codec/context for each frame (first is already ok).

     * Each frame is 1 or 2 channels - up to 5 frames allowed

     */

    for (i = 1; i < s->frames; i++) {

        s->mp3decctx[i] = av_mallocz(sizeof(MPADecodeContext));

        s->mp3decctx[i]->compute_antialias = s->mp3decctx[0]->compute_antialias;

        s->mp3decctx[i]->adu_mode = 1;

        s->mp3decctx[i]->avctx = avctx;

    }



    return 0;

}
