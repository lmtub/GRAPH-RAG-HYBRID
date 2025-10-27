static void sdp_parse_fmtp(AVStream *st, const char *p)

{

    char attr[256];

    /* Vorbis setup headers can be up to 12KB and are sent base64

     * encoded, giving a 12KB * (4/3) = 16KB FMTP line. */

    char value[16384];

    int i;



    RTSPStream *rtsp_st = st->priv_data;

    AVCodecContext *codec = st->codec;

    RTPPayloadData *rtp_payload_data = &rtsp_st->rtp_payload_data;



    /* loop on each attribute */

    while(rtsp_next_attr_and_value(&p, attr, sizeof(attr), value, sizeof(value)))

    {

        /* grab the codec extra_data from the config parameter of the fmtp line */

        sdp_parse_fmtp_config(codec, rtsp_st->dynamic_protocol_context,

                              attr, value);

        /* Looking for a known attribute */

        for (i = 0; attr_names[i].str; ++i) {

            if (!strcasecmp(attr, attr_names[i].str)) {

                if (attr_names[i].type == ATTR_NAME_TYPE_INT)

                    *(int *)((char *)rtp_payload_data + attr_names[i].offset) = atoi(value);

                else if (attr_names[i].type == ATTR_NAME_TYPE_STR)

                    *(char **)((char *)rtp_payload_data + attr_names[i].offset) = av_strdup(value);

            }

        }

    }

}
