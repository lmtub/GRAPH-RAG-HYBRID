static int mpegts_push_data(MpegTSFilter *filter,

                            const uint8_t *buf, int buf_size, int is_start,

                            int64_t pos)

{

    PESContext *pes = filter->u.pes_filter.opaque;

    MpegTSContext *ts = pes->ts;

    const uint8_t *p;

    int len, code;



    if(!ts->pkt)

        return 0;



    if (is_start) {

        if (pes->state == MPEGTS_PAYLOAD && pes->data_index > 0) {

            new_pes_packet(pes, ts->pkt);

            ts->stop_parse = 1;

        }

        pes->state = MPEGTS_HEADER;

        pes->data_index = 0;

        pes->ts_packet_pos = pos;

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

                av_hex_dump_log(pes->stream, AV_LOG_DEBUG, pes->header, pes->data_index);

#endif

                if (pes->header[0] == 0x00 && pes->header[1] == 0x00 &&

                    pes->header[2] == 0x01) {

                    /* it must be an mpeg2 PES stream */

                    code = pes->header[3] | 0x100;

                    dprintf(pes->stream, "pid=%x pes_code=%#x\n", pes->pid, code);



                    if ((pes->st && pes->st->discard == AVDISCARD_ALL) ||

                        code == 0x1be) /* padding_stream */

                        goto skip;



                    /* stream not present in PMT */

                    if (!pes->st)

                        pes->st = new_pes_av_stream(pes, 0, code);

                    if (!pes->st)

                        return AVERROR(ENOMEM);



                    pes->total_size = AV_RB16(pes->header + 4);

                    /* NOTE: a zero total size means the PES size is

                       unbounded */

                    if (!pes->total_size)

                        pes->total_size = MAX_PES_PAYLOAD;



                    /* allocate pes buffer */

                    pes->buffer = av_malloc(pes->total_size+FF_INPUT_BUFFER_PADDING_SIZE);

                    if (!pes->buffer)

                        return AVERROR(ENOMEM);



                    if (code != 0x1bc && code != 0x1bf && /* program_stream_map, private_stream_2 */

                        code != 0x1f0 && code != 0x1f1 && /* ECM, EMM */

                        code != 0x1ff && code != 0x1f2 && /* program_stream_directory, DSMCC_stream */

                        code != 0x1f8) {                  /* ITU-T Rec. H.222.1 type E stream */

                        pes->state = MPEGTS_PESHEADER_FILL;

                        pes->pes_header_size = pes->header[8] + 9;

                    } else {

                        pes->state = MPEGTS_PAYLOAD;

                        pes->data_index = 0;

                    }

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

            if (len < 0)

                return -1;

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

                    pes->dts = pes->pts = get_pts(r);

                    r += 5;

                } else if ((flags & 0xc0) == 0xc0) {

                    pes->pts = get_pts(r);

                    r += 5;

                    pes->dts = get_pts(r);

                    r += 5;

                }



                /* we got the full header. We parse it and get the payload */

                pes->state = MPEGTS_PAYLOAD;

                pes->data_index = 0;

            }

            break;

        case MPEGTS_PAYLOAD:

            if (buf_size > 0) {

                if (pes->data_index+buf_size > pes->total_size) {

                    new_pes_packet(pes, ts->pkt);

                    pes->total_size = MAX_PES_PAYLOAD;

                    pes->buffer = av_malloc(pes->total_size+FF_INPUT_BUFFER_PADDING_SIZE);

                    if (!pes->buffer)

                        return AVERROR(ENOMEM);

                    ts->stop_parse = 1;

                }

                memcpy(pes->buffer+pes->data_index, p, buf_size);

                pes->data_index += buf_size;

            }

            buf_size = 0;

            break;

        case MPEGTS_SKIP:

            buf_size = 0;

            break;

        }

    }



    return 0;

}
