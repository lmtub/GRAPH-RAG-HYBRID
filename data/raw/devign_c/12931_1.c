static int asf_read_payload(AVFormatContext *s, AVPacket *pkt)

{

    ASFContext *asf = s->priv_data;

    AVIOContext *pb = s->pb;

    int ret, i;

    ASFPacket *asf_pkt = NULL;



    if (!asf->sub_left) {

        uint32_t off_len, media_len;

        uint8_t stream_num;



        stream_num = avio_r8(pb);

        asf->stream_index = stream_num & ASF_STREAM_NUM;

        for (i = 0; i < asf->nb_streams; i++) {

            if (asf->stream_index == asf->asf_st[i]->stream_index) {

                asf_pkt               = &asf->asf_st[i]->pkt;

                asf_pkt->stream_index = asf->asf_st[i]->index;

                break;

            }

        }

        if (!asf_pkt)

            return AVERROR_INVALIDDATA;

        if (stream_num >> 7)

            asf_pkt->flags |= AV_PKT_FLAG_KEY;

        READ_LEN(asf->prop_flags & ASF_PL_MASK_MEDIA_OBJECT_NUMBER_LENGTH_FIELD_SIZE,

                 ASF_PL_FLAG_MEDIA_OBJECT_NUMBER_LENGTH_FIELD_, media_len);

        READ_LEN(asf->prop_flags & ASF_PL_MASK_OFFSET_INTO_MEDIA_OBJECT_LENGTH_FIELD_SIZE,

                 ASF_PL_FLAG_OFFSET_INTO_MEDIA_OBJECT_LENGTH_FIELD_, off_len);

        READ_LEN(asf->prop_flags & ASF_PL_MASK_REPLICATED_DATA_LENGTH_FIELD_SIZE,

                 ASF_PL_FLAG_REPLICATED_DATA_LENGTH_FIELD_, asf->rep_data_len);

        if (asf_pkt->size_left && (asf_pkt->frame_num != media_len)) {

            av_log(s, AV_LOG_WARNING, "Unfinished frame will be ignored\n");

            reset_packet(asf_pkt);

        }

        asf_pkt->frame_num = media_len;

        asf->sub_dts = off_len;

        if (asf->nb_mult_left) {

            if ((ret = asf_read_multiple_payload(s, pkt, asf_pkt)) < 0)

                return ret;

        } else if (asf->rep_data_len == 1) {

            asf->sub_left = 1;

            asf->state    = READ_SINGLE;

            pkt->flags    = asf_pkt->flags;

            if ((ret = asf_read_subpayload(s, pkt, 1)) < 0)

                return ret;

        } else {

            if ((ret = asf_read_single_payload(s, pkt, asf_pkt)) < 0)

                return ret;

        }

    } else {

        for (i = 0; i <= asf->nb_streams; i++) {

            if (asf->stream_index == asf->asf_st[i]->stream_index) {

                asf_pkt = &asf->asf_st[i]->pkt;

                break;

            }

        }

        if (!asf_pkt)

            return AVERROR_INVALIDDATA;

        pkt->flags         = asf_pkt->flags;

        pkt->dts           = asf_pkt->dts;

        pkt->stream_index  = asf->asf_st[i]->index;

        if ((ret = asf_read_subpayload(s, pkt, 0)) < 0) // read subpayload without its header

            return ret;

    }



    return 0;

}
