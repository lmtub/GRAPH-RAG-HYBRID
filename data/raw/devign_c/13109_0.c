rtsp_read_reply (AVFormatContext *s, RTSPMessageHeader *reply,

                 unsigned char **content_ptr, int return_on_interleaved_data)

{

    RTSPState *rt = s->priv_data;

    char buf[4096], buf1[1024], *q;

    unsigned char ch;

    const char *p;

    int ret, content_length, line_count = 0;

    unsigned char *content = NULL;



    memset(reply, 0, sizeof(*reply));



    /* parse reply (XXX: use buffers) */

    rt->last_reply[0] = '\0';

    for(;;) {

        q = buf;

        for(;;) {

            ret = url_read_complete(rt->rtsp_hd, &ch, 1);

#ifdef DEBUG_RTP_TCP

            dprintf(s, "ret=%d c=%02x [%c]\n", ret, ch, ch);

#endif

            if (ret != 1)

                return -1;

            if (ch == '\n')

                break;

            if (ch == '$') {

                /* XXX: only parse it if first char on line ? */

                if (return_on_interleaved_data) {

                    return 1;

                } else

                rtsp_skip_packet(s);

            } else if (ch != '\r') {

                if ((q - buf) < sizeof(buf) - 1)

                    *q++ = ch;

            }

        }

        *q = '\0';



        dprintf(s, "line='%s'\n", buf);



        /* test if last line */

        if (buf[0] == '\0')

            break;

        p = buf;

        if (line_count == 0) {

            /* get reply code */

            get_word(buf1, sizeof(buf1), &p);

            get_word(buf1, sizeof(buf1), &p);

            reply->status_code = atoi(buf1);

        } else {

            rtsp_parse_line(reply, p);

            av_strlcat(rt->last_reply, p,    sizeof(rt->last_reply));

            av_strlcat(rt->last_reply, "\n", sizeof(rt->last_reply));

        }

        line_count++;

    }



    if (rt->session_id[0] == '\0' && reply->session_id[0] != '\0')

        av_strlcpy(rt->session_id, reply->session_id, sizeof(rt->session_id));



    content_length = reply->content_length;

    if (content_length > 0) {

        /* leave some room for a trailing '\0' (useful for simple parsing) */

        content = av_malloc(content_length + 1);

        (void)url_read_complete(rt->rtsp_hd, content, content_length);

        content[content_length] = '\0';

    }

    if (content_ptr)

        *content_ptr = content;

    else

        av_free(content);



    /* EOS */

    if (reply->notice == 2101 /* End-of-Stream Reached */      ||

        reply->notice == 2104 /* Start-of-Stream Reached */    ||

        reply->notice == 2306 /* Continuous Feed Terminated */)

        rt->state = RTSP_STATE_IDLE;

    else if (reply->notice >= 4400 && reply->notice < 5500)

        return AVERROR(EIO); /* data or server error */

    else if (reply->notice == 2401 /* Ticket Expired */ ||

             (reply->notice >= 5500 && reply->notice < 5600) /* end of term */ )

        return AVERROR(EPERM);



    return 0;

}
