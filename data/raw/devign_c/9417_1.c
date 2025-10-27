int ff_vbv_update(MpegEncContext *s, int frame_size)

{

    RateControlContext *rcc = &s->rc_context;

    const double fps        = get_fps(s->avctx);

    const int buffer_size   = s->avctx->rc_buffer_size;

    const double min_rate   = s->avctx->rc_min_rate / fps;

    const double max_rate   = s->avctx->rc_max_rate / fps;



    av_dlog(s, "%d %f %d %f %f\n",

            buffer_size, rcc->buffer_index, frame_size, min_rate, max_rate);



    if (buffer_size) {

        int left;



        rcc->buffer_index -= frame_size;

        if (rcc->buffer_index < 0) {

            av_log(s->avctx, AV_LOG_ERROR, "rc buffer underflow\n");




            rcc->buffer_index = 0;




        left = buffer_size - rcc->buffer_index - 1;

        rcc->buffer_index += av_clip(left, min_rate, max_rate);



        if (rcc->buffer_index > buffer_size) {

            int stuffing = ceil((rcc->buffer_index - buffer_size) / 8);



            if (stuffing < 4 && s->codec_id == AV_CODEC_ID_MPEG4)

                stuffing = 4;

            rcc->buffer_index -= 8 * stuffing;



            if (s->avctx->debug & FF_DEBUG_RC)

                av_log(s->avctx, AV_LOG_DEBUG, "stuffing %d bytes\n", stuffing);



            return stuffing;



    return 0;
