static void sdp_parse_line(AVFormatContext *s, SDPParseState *s1,

                           int letter, const char *buf)

{

    RTSPState *rt = s->priv_data;

    char buf1[64], st_type[64];

    const char *p;

    int codec_type, payload_type, i;

    AVStream *st;

    RTSPStream *rtsp_st;

    struct in_addr sdp_ip;

    int ttl;



#ifdef DEBUG

    printf("sdp: %c='%s'\n", letter, buf);

#endif



    p = buf;

    switch(letter) {

    case 'c':

        get_word(buf1, sizeof(buf1), &p);

        if (strcmp(buf1, "IN") != 0)

            return;

        get_word(buf1, sizeof(buf1), &p);

        if (strcmp(buf1, "IP4") != 0)

            return;

        get_word_sep(buf1, sizeof(buf1), "/", &p);

        if (inet_aton(buf1, &sdp_ip) == 0)

            return;

        ttl = 16;

        if (*p == '/') {

            p++;

            get_word_sep(buf1, sizeof(buf1), "/", &p);

            ttl = atoi(buf1);

        }

        if (s->nb_streams == 0) {

            s1->default_ip = sdp_ip;

            s1->default_ttl = ttl;

        } else {

            st = s->streams[s->nb_streams - 1];

            rtsp_st = st->priv_data;

            rtsp_st->sdp_ip = sdp_ip;

            rtsp_st->sdp_ttl = ttl;

        }

        break;

    case 's':

        pstrcpy(s->title, sizeof(s->title), p);

        break;

    case 'i':

        if (s->nb_streams == 0) {

            pstrcpy(s->comment, sizeof(s->comment), p);

            break;

        }

        break;

    case 'm':

        /* new stream */

        get_word(st_type, sizeof(st_type), &p);

        if (!strcmp(st_type, "audio")) {

            codec_type = CODEC_TYPE_AUDIO;

        } else if (!strcmp(st_type, "video")) {

            codec_type = CODEC_TYPE_VIDEO;

        } else {

            return;

        }

        rtsp_st = av_mallocz(sizeof(RTSPStream));

        if (!rtsp_st)

            return;

        rtsp_st->stream_index = -1;

        dynarray_add(&rt->rtsp_streams, &rt->nb_rtsp_streams, rtsp_st);



        rtsp_st->sdp_ip = s1->default_ip;

        rtsp_st->sdp_ttl = s1->default_ttl;



        get_word(buf1, sizeof(buf1), &p); /* port */

        rtsp_st->sdp_port = atoi(buf1);



        get_word(buf1, sizeof(buf1), &p); /* protocol (ignored) */

        

        /* XXX: handle list of formats */

        get_word(buf1, sizeof(buf1), &p); /* format list */

        rtsp_st->sdp_payload_type = atoi(buf1);



        if (rtsp_st->sdp_payload_type == RTP_PT_MPEG2TS) {

            /* no corresponding stream */

        } else {

            st = av_new_stream(s, 0);

            if (!st)

                return;

            st->priv_data = rtsp_st;

            rtsp_st->stream_index = st->index;

            st->codec.codec_type = codec_type;

            if (rtsp_st->sdp_payload_type < 96) {

                /* if standard payload type, we can find the codec right now */

                rtp_get_codec_info(&st->codec, rtsp_st->sdp_payload_type);

            }

        }

        /* put a default control url */

        pstrcpy(rtsp_st->control_url, sizeof(rtsp_st->control_url), s->filename);

        break;

    case 'a':

        if (strstart(p, "control:", &p) && s->nb_streams > 0) {

            char proto[32];

            /* get the control url */

            st = s->streams[s->nb_streams - 1];

            rtsp_st = st->priv_data;

            

            /* XXX: may need to add full url resolution */

            url_split(proto, sizeof(proto), NULL, 0, NULL, NULL, 0, p);

            if (proto[0] == '\0') {

                /* relative control URL */

                pstrcat(rtsp_st->control_url, sizeof(rtsp_st->control_url), "/");

                pstrcat(rtsp_st->control_url, sizeof(rtsp_st->control_url), p);

            } else {

                pstrcpy(rtsp_st->control_url, sizeof(rtsp_st->control_url), p);

            }

        } else if (strstart(p, "rtpmap:", &p)) {

            /* NOTE: rtpmap is only supported AFTER the 'm=' tag */

            get_word(buf1, sizeof(buf1), &p); 

            payload_type = atoi(buf1);

            for(i = 0; i < s->nb_streams;i++) {

                st = s->streams[i];

                rtsp_st = st->priv_data;

                if (rtsp_st->sdp_payload_type == payload_type) {

                    sdp_parse_rtpmap(&st->codec, p);

                }

            }

        } else if (strstart(p, "fmtp:", &p)) {

            /* NOTE: fmtp is only supported AFTER the 'a=rtpmap:xxx' tag */

            get_word(buf1, sizeof(buf1), &p); 

            payload_type = atoi(buf1);

            for(i = 0; i < s->nb_streams;i++) {

                st = s->streams[i];

                rtsp_st = st->priv_data;

                if (rtsp_st->sdp_payload_type == payload_type) {

                    sdp_parse_fmtp(&st->codec, p);

                }

            }

        }

        break;

    }

}
