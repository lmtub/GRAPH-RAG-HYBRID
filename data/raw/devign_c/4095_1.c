static void pkt_dump_internal(void *avcl, FILE *f, int level, const AVPacket *pkt,

                              int dump_payload, AVRational time_base)

{

    HEXDUMP_PRINT("stream #%d:\n", pkt->stream_index);

    HEXDUMP_PRINT("  keyframe=%d\n", (pkt->flags & AV_PKT_FLAG_KEY) != 0);

    HEXDUMP_PRINT("  duration=%0.3f\n", pkt->duration * av_q2d(time_base));

    /* DTS is _always_ valid after av_read_frame() */

    HEXDUMP_PRINT("  dts=");

    if (pkt->dts == AV_NOPTS_VALUE)

        HEXDUMP_PRINT("N/A");

    else

        HEXDUMP_PRINT("%0.3f", pkt->dts * av_q2d(time_base));

    /* PTS may not be known if B-frames are present. */

    HEXDUMP_PRINT("  pts=");

    if (pkt->pts == AV_NOPTS_VALUE)

        HEXDUMP_PRINT("N/A");

    else

        HEXDUMP_PRINT("%0.3f", pkt->pts * av_q2d(time_base));

    HEXDUMP_PRINT("\n");

    HEXDUMP_PRINT("  size=%d\n", pkt->size);

    if (dump_payload)

        av_hex_dump(f, pkt->data, pkt->size);

}
