static void mpegts_push_data(void *opaque,

                             const uint8_t *buf, int buf_size, int is_start)

{

    PESContext *pes = opaque;

    MpegTSContext *ts = pes->stream->priv_data;

    AVStream *st;

    const uint8_t *p;

    int len, code, codec_type, codec_id;

    

    if (is_start) {

        pes->state = MPEGTS_HEADER;

        pes->data_index = 0;

    }

    p = buf;

    while (buf_size > 0) {

        switch(pes->state) {

        case MPEGTS_HEADER:

            len = PES_START_SIZE - pes->data_index;

            if (len > buf_size)

                len = buf_size;

            memcpy(pes->header + pes->data_index, p, len);

            pes->data_index += len;

            p += len;

            buf_size -= len;

            if (pes->data_index == PES_START_SIZE) {

                /* we got all the PES or section header. We can now

                   decide */

#if 0

                av_hex_dump(pes->header, pes->data_index);

#endif

                if (pes->header[0] == 0x00 && pes->header[1] == 0x00 &&

                    pes->header[2] == 0x01) {

                    /* it must be an mpeg2 PES stream */

                    /* XXX: add AC3 support */

                    code = pes->header[3] | 0x100;

                    if (!((code >= 0x1c0 && code <= 0x1df) ||

                          (code >= 0x1e0 && code <= 0x1ef)))

                        goto skip;

                    if (!pes->st) {

                        /* allocate stream */

                        if (code >= 0x1c0 && code <= 0x1df) {

                            codec_type = CODEC_TYPE_AUDIO;

                            codec_id = CODEC_ID_MP2;

                        } else {

                            codec_type = CODEC_TYPE_VIDEO;

                            codec_id = CODEC_ID_MPEG1VIDEO;

                        }

                        st = av_new_stream(pes->stream, pes->pid);

                        if (st) {

                            st->priv_data = pes;

                            st->codec.codec_type = codec_type;

                            st->codec.codec_id = codec_id;

                            pes->st = st;

                        }

                    }

                    pes->state = MPEGTS_PESHEADER_FILL;

                    pes->total_size = (pes->header[4] << 8) | pes->header[5];

                    /* NOTE: a zero total size means the PES size is

                       unbounded */

                    if (pes->total_size)

                        pes->total_size += 6;

                    pes->pes_header_size = pes->header[8] + 9;

                } else {

                    /* otherwise, it should be a table */

                    /* skip packet */

                skip:

                    pes->state = MPEGTS_SKIP;

                    continue;

                }

            }

            break;

            /**********************************************/

            /* PES packing parsing */

        case MPEGTS_PESHEADER_FILL:

            len = pes->pes_header_size - pes->data_index;

            if (len > buf_size)

                len = buf_size;

            memcpy(pes->header + pes->data_index, p, len);

            pes->data_index += len;

            p += len;

            buf_size -= len;

            if (pes->data_index == pes->pes_header_size) {

                const uint8_t *r;

                unsigned int flags;



                flags = pes->header[7];

                r = pes->header + 9;

                pes->pts = AV_NOPTS_VALUE;

                pes->dts = AV_NOPTS_VALUE;

                if ((flags & 0xc0) == 0x80) {

                    pes->pts = get_pts(r);

                    r += 5;

                } else if ((flags & 0xc0) == 0xc0) {

                    pes->pts = get_pts(r);

                    r += 5;

                    pes->dts = get_pts(r);

                    r += 5;

                }

                /* we got the full header. We parse it and get the payload */

                pes->state = MPEGTS_PAYLOAD;

            }

            break;

        case MPEGTS_PAYLOAD:

            if (pes->total_size) {

                len = pes->total_size - pes->data_index;

                if (len > buf_size)

                    len = buf_size;

            } else {

                len = buf_size;

            }

            if (len > 0) {

                AVPacket *pkt = ts->pkt;

                if (pes->st && av_new_packet(pkt, len) == 0) {

                    memcpy(pkt->data, p, len);

                    pkt->stream_index = pes->st->index;

                    pkt->pts = pes->pts;

                    /* reset pts values */

                    pes->pts = AV_NOPTS_VALUE;

                    pes->dts = AV_NOPTS_VALUE;

                    ts->stop_parse = 1;

                    return;

                }

            }

            buf_size = 0;

            break;

        case MPEGTS_SKIP:

            buf_size = 0;

            break;

        }

    }

}
