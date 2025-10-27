void rtp_parse_close(RTPDemuxContext *s)

{

    // TODO: fold this into the protocol specific data fields.



    if (!strcmp(ff_rtp_enc_name(s->payload_type), "MP2T")) {

        ff_mpegts_parse_close(s->ts);

    }

    av_free(s);

}