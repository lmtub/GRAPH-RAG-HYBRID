static void reset_packet_state(AVFormatContext *s)

{

    ASFContext *asf        = s->priv_data;

    int i;



    asf->state             = PARSE_PACKET_HEADER;

    asf->offset            = 0;

    asf->return_subpayload = 0;

    asf->sub_left          = 0;

    asf->sub_header_offset = 0;

    asf->packet_offset     = asf->first_packet_offset;

    asf->pad_len           = 0;

    asf->rep_data_len      = 0;

    asf->dts_delta         = 0;

    asf->mult_sub_len      = 0;

    asf->nb_mult_left      = 0;

    asf->nb_sub            = 0;

    asf->prop_flags        = 0;

    asf->sub_dts           = 0;

    asf->dts               = 0;

    for (i = 0; i < asf->nb_streams; i++) {

        ASFPacket *pkt = &asf->asf_st[i]->pkt;

        pkt->size_left = 0;

        pkt->data_size = 0;

        pkt->duration  = 0;

        pkt->flags     = 0;

        pkt->dts       = 0;

        pkt->duration  = 0;

        av_free_packet(&pkt->avpkt);

        av_init_packet(&pkt->avpkt);

    }

}
