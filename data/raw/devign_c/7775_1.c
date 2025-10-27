static int zerocodec_decode_frame(AVCodecContext *avctx, void *data,

                                  int *data_size, AVPacket *avpkt)

{

    ZeroCodecContext *zc = avctx->priv_data;

    AVFrame *pic         = avctx->coded_frame;

    AVFrame *prev_pic    = &zc->previous_frame;

    z_stream *zstream    = &zc->zstream;

    uint8_t *prev, *dst;

    int i, j, zret;



    pic->reference = 3;



    if (avctx->get_buffer(avctx, pic) < 0) {

        av_log(avctx, AV_LOG_ERROR, "Could not allocate buffer.\n");

        return AVERROR(ENOMEM);




    zret = inflateReset(zstream);



    if (zret != Z_OK) {

        av_log(avctx, AV_LOG_ERROR, "Could not reset inflate: %d\n", zret);

        return AVERROR(EINVAL);




    zstream->next_in   = avpkt->data;

    zstream->avail_in  = avpkt->size;



    prev = prev_pic->data[0];

    dst  = pic->data[0];



    /**

     * ZeroCodec has very simple interframe compression. If a value

     * is the same as the previous frame, set it to 0.

     */



    if (avpkt->flags & AV_PKT_FLAG_KEY) {

        pic->key_frame = 1;

        pic->pict_type = AV_PICTURE_TYPE_I;

    } else {





        pic->key_frame = 0;

        pic->pict_type = AV_PICTURE_TYPE_P;




    for (i = 0; i < avctx->height; i++) {

        zstream->next_out  = dst;

        zstream->avail_out = avctx->width << 1;

        zret = inflate(zstream, Z_SYNC_FLUSH);

        if (zret != Z_OK && zret != Z_STREAM_END) {

            av_log(avctx, AV_LOG_ERROR,

                   "Inflate failed with return code: %d\n", zret);

            return AVERROR(EINVAL);




        if (!(avpkt->flags & AV_PKT_FLAG_KEY))

            for (j = 0; j < avctx->width << 1; j++)

                dst[j] += prev[j] & -!dst[j];



        prev += prev_pic->linesize[0];

        dst  += pic->linesize[0];




    /* Release the previous buffer if need be */

    if (prev_pic->data[0])

        avctx->release_buffer(avctx, prev_pic);



    /* Store the previouse frame for use later */

    *prev_pic = *pic;



    *data_size = sizeof(AVFrame);

    *(AVFrame *)data = *pic;



    return avpkt->size;
