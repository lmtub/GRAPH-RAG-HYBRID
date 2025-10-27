static int hevc_handle_packet(AVFormatContext *ctx, PayloadContext *rtp_hevc_ctx,

                              AVStream *st, AVPacket *pkt, uint32_t *timestamp,

                              const uint8_t *buf, int len, uint16_t seq,

                              int flags)

{

    const uint8_t *rtp_pl = buf;

    int tid, lid, nal_type;

    int first_fragment, last_fragment, fu_type;

    uint8_t new_nal_header[2];

    int res = 0;



    /* sanity check for size of input packet: 1 byte payload at least */

    if (len < RTP_HEVC_PAYLOAD_HEADER_SIZE + 1) {

        av_log(ctx, AV_LOG_ERROR, "Too short RTP/HEVC packet, got %d bytes\n", len);

        return AVERROR_INVALIDDATA;

    }



    /*

     * decode the HEVC payload header according to section 4 of draft version 6:

     *

     *    0                   1

     *    0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5

     *   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

     *   |F|   Type    |  LayerId  | TID |

     *   +-------------+-----------------+

     *

     *      Forbidden zero (F): 1 bit

     *      NAL unit type (Type): 6 bits

     *      NUH layer ID (LayerId): 6 bits

     *      NUH temporal ID plus 1 (TID): 3 bits

     */

    nal_type =  (buf[0] >> 1) & 0x3f;

    lid  = ((buf[0] << 5) & 0x20) | ((buf[1] >> 3) & 0x1f);

    tid  =   buf[1] & 0x07;



    /* sanity check for correct layer ID */

    if (lid) {

        /* future scalable or 3D video coding extensions */

        avpriv_report_missing_feature(ctx, "Multi-layer HEVC coding\n");

        return AVERROR_PATCHWELCOME;

    }



    /* sanity check for correct temporal ID */

    if (!tid) {

        av_log(ctx, AV_LOG_ERROR, "Illegal temporal ID in RTP/HEVC packet\n");

        return AVERROR_INVALIDDATA;

    }



    /* sanity check for correct NAL unit type */

    if (nal_type > 50) {

        av_log(ctx, AV_LOG_ERROR, "Unsupported (HEVC) NAL type (%d)\n", nal_type);

        return AVERROR_INVALIDDATA;

    }



    switch (nal_type) {

    /* video parameter set (VPS) */

    case 32:

    /* sequence parameter set (SPS) */

    case 33:

    /* picture parameter set (PPS) */

    case 34:

    /*  supplemental enhancement information (SEI) */

    case 39:

    /* single NAL unit packet */

    default:

        /* sanity check for size of input packet: 1 byte payload at least */

        if (len < 1) {

            av_log(ctx, AV_LOG_ERROR,

                   "Too short RTP/HEVC packet, got %d bytes of NAL unit type %d\n",

                   len, nal_type);

            return AVERROR_INVALIDDATA;

        }



        /* create A/V packet */

        if ((res = av_new_packet(pkt, sizeof(start_sequence) + len)) < 0)

            return res;

        /* A/V packet: copy start sequence */

        memcpy(pkt->data, start_sequence, sizeof(start_sequence));

        /* A/V packet: copy NAL unit data */

        memcpy(pkt->data + sizeof(start_sequence), buf, len);



        break;

    /* aggregated packet (AP) - with two or more NAL units */

    case 48:

        /* pass the HEVC payload header */

        buf += RTP_HEVC_PAYLOAD_HEADER_SIZE;

        len -= RTP_HEVC_PAYLOAD_HEADER_SIZE;



        /* pass the HEVC DONL field */

        if (rtp_hevc_ctx->using_donl_field) {

            buf += RTP_HEVC_DONL_FIELD_SIZE;

            len -= RTP_HEVC_DONL_FIELD_SIZE;

        }



        res = ff_h264_handle_aggregated_packet(ctx, pkt, buf, len,

                                               rtp_hevc_ctx->using_donl_field ?

                                               RTP_HEVC_DOND_FIELD_SIZE : 0,

                                               NULL, 0);

        if (res < 0)

            return res;

        break;

    /* fragmentation unit (FU) */

    case 49:

        /* pass the HEVC payload header */

        buf += RTP_HEVC_PAYLOAD_HEADER_SIZE;

        len -= RTP_HEVC_PAYLOAD_HEADER_SIZE;



        /*

         *    decode the FU header

         *

         *     0 1 2 3 4 5 6 7

         *    +-+-+-+-+-+-+-+-+

         *    |S|E|  FuType   |

         *    +---------------+

         *

         *       Start fragment (S): 1 bit

         *       End fragment (E): 1 bit

         *       FuType: 6 bits

         */

        first_fragment = buf[0] & 0x80;

        last_fragment  = buf[0] & 0x40;

        fu_type        = buf[0] & 0x3f;



        /* pass the HEVC FU header */

        buf += RTP_HEVC_FU_HEADER_SIZE;

        len -= RTP_HEVC_FU_HEADER_SIZE;



        /* pass the HEVC DONL field */

        if (rtp_hevc_ctx->using_donl_field) {

            buf += RTP_HEVC_DONL_FIELD_SIZE;

            len -= RTP_HEVC_DONL_FIELD_SIZE;

        }



        av_dlog(ctx, " FU type %d with %d bytes\n", fu_type, len);



        if (len <= 0) {

            /* sanity check for size of input packet: 1 byte payload at least */

            av_log(ctx, AV_LOG_ERROR,

                   "Too short RTP/HEVC packet, got %d bytes of NAL unit type %d\n",

                   len, nal_type);

            return AVERROR_INVALIDDATA;

        }



        if (first_fragment && last_fragment) {

            av_log(ctx, AV_LOG_ERROR, "Illegal combination of S and E bit in RTP/HEVC packet\n");

            return AVERROR_INVALIDDATA;

        }



        new_nal_header[0] = (rtp_pl[0] & 0x81) | (fu_type << 1);

        new_nal_header[1] = rtp_pl[1];



        res = ff_h264_handle_frag_packet(pkt, buf, len, first_fragment,

                                         new_nal_header, sizeof(new_nal_header));



        break;

    /* PACI packet */

    case 50:

        /* Temporal scalability control information (TSCI) */

        avpriv_report_missing_feature(ctx, "PACI packets for RTP/HEVC\n");

        res = AVERROR_PATCHWELCOME;

        break;

    }



    pkt->stream_index = st->index;



    return res;

}
