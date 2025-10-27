static void show_packet(AVFormatContext *fmt_ctx, AVPacket *pkt)

{

    char val_str[128];

    AVStream *st = fmt_ctx->streams[pkt->stream_index];



    printf("[PACKET]\n");

    printf("codec_type=%s\n", media_type_string(st->codec->codec_type));

    printf("stream_index=%d\n", pkt->stream_index);

    printf("pts=%s\n", ts_value_string(val_str, sizeof(val_str), pkt->pts));

    printf("pts_time=%s\n", time_value_string(val_str, sizeof(val_str),

                                              pkt->pts, &st->time_base));

    printf("dts=%s\n", ts_value_string(val_str, sizeof(val_str), pkt->dts));

    printf("dts_time=%s\n", time_value_string(val_str, sizeof(val_str),

                                              pkt->dts, &st->time_base));

    printf("duration=%s\n", ts_value_string(val_str, sizeof(val_str),

                                            pkt->duration));

    printf("duration_time=%s\n", time_value_string(val_str, sizeof(val_str),

                                                   pkt->duration,

                                                   &st->time_base));

    printf("size=%s\n", value_string(val_str, sizeof(val_str),

                                     pkt->size, unit_byte_str));

    printf("pos=%"PRId64"\n", pkt->pos);

    printf("flags=%c\n", pkt->flags & AV_PKT_FLAG_KEY ? 'K' : '_');

    printf("[/PACKET]\n");

}
