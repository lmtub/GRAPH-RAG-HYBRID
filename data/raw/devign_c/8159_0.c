static int decode_fctl_chunk(AVFormatContext *s, APNGDemuxContext *ctx, AVPacket *pkt)

{

    uint32_t sequence_number, width, height, x_offset, y_offset;

    uint16_t delay_num, delay_den;

    uint8_t dispose_op, blend_op;



    sequence_number = avio_rb32(s->pb);

    width           = avio_rb32(s->pb);

    height          = avio_rb32(s->pb);

    x_offset        = avio_rb32(s->pb);

    y_offset        = avio_rb32(s->pb);

    delay_num       = avio_rb16(s->pb);

    delay_den       = avio_rb16(s->pb);

    dispose_op      = avio_r8(s->pb);

    blend_op        = avio_r8(s->pb);

    avio_skip(s->pb, 4); /* crc */



    /* default is hundredths of seconds */

    if (!delay_den)

        delay_den = 100;

    if (!delay_num || delay_den / delay_num > ctx->max_fps) {

        delay_num = 1;

        delay_den = ctx->default_fps;

    }

    ctx->pkt_duration = av_rescale_q(delay_num,

                                     (AVRational){ 1, delay_den },

                                     s->streams[0]->time_base);



    av_log(s, AV_LOG_DEBUG, "%s: "

            "sequence_number: %"PRId32", "

            "width: %"PRIu32", "

            "height: %"PRIu32", "

            "x_offset: %"PRIu32", "

            "y_offset: %"PRIu32", "

            "delay_num: %"PRIu16", "

            "delay_den: %"PRIu16", "

            "dispose_op: %d, "

            "blend_op: %d\n",

            __FUNCTION__,

            sequence_number,

            width,

            height,

            x_offset,

            y_offset,

            delay_num,

            delay_den,

            dispose_op,

            blend_op);



    if (width != s->streams[0]->codecpar->width ||

        height != s->streams[0]->codecpar->height ||

        x_offset != 0 ||

        y_offset != 0) {

        if (sequence_number == 0 ||

            x_offset >= s->streams[0]->codecpar->width ||

            width > s->streams[0]->codecpar->width - x_offset ||

            y_offset >= s->streams[0]->codecpar->height ||

            height > s->streams[0]->codecpar->height - y_offset)

            return AVERROR_INVALIDDATA;

        ctx->is_key_frame = 0;

    } else {

        if (sequence_number == 0 && dispose_op == APNG_DISPOSE_OP_PREVIOUS)

            dispose_op = APNG_DISPOSE_OP_BACKGROUND;

        ctx->is_key_frame = dispose_op == APNG_DISPOSE_OP_BACKGROUND ||

                            blend_op   == APNG_BLEND_OP_SOURCE;

    }



    return 0;

}
