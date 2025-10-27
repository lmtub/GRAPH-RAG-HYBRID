static void rtsp_send_cmd(AVFormatContext *s, 

                          const char *cmd, RTSPHeader *reply, 

                          unsigned char **content_ptr)

{

    RTSPState *rt = s->priv_data;

    char buf[4096], buf1[1024], *q;

    unsigned char ch;

    const char *p;

    int content_length, line_count;

    unsigned char *content = NULL;



    memset(reply, 0, sizeof(RTSPHeader));



    rt->seq++;

    pstrcpy(buf, sizeof(buf), cmd);

    snprintf(buf1, sizeof(buf1), "CSeq: %d\r\n", rt->seq);

    pstrcat(buf, sizeof(buf), buf1);

    if (rt->session_id[0] != '\0' && !strstr(cmd, "\nIf-Match:")) {

        snprintf(buf1, sizeof(buf1), "Session: %s\r\n", rt->session_id);

        pstrcat(buf, sizeof(buf), buf1);

    }

    pstrcat(buf, sizeof(buf), "\r\n");

#ifdef DEBUG

    printf("Sending:\n%s--\n", buf);

#endif

    url_write(rt->rtsp_hd, buf, strlen(buf));



    /* parse reply (XXX: use buffers) */

    line_count = 0;

    rt->last_reply[0] = '\0';

    for(;;) {

        q = buf;

        for(;;) {

            if (url_read(rt->rtsp_hd, &ch, 1) == 0)

                break;

            if (ch == '\n')

                break;

            if (ch != '\r') {

                if ((q - buf) < sizeof(buf) - 1)

                    *q++ = ch;

            }

        }

        *q = '\0';

#ifdef DEBUG

        printf("line='%s'\n", buf);

#endif

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

            pstrcat(rt->last_reply, sizeof(rt->last_reply), p);

            pstrcat(rt->last_reply, sizeof(rt->last_reply), "\n");

        }

        line_count++;

    }

    

    if (rt->session_id[0] == '\0' && reply->session_id[0] != '\0')

        pstrcpy(rt->session_id, sizeof(rt->session_id), reply->session_id);

    

    content_length = reply->content_length;

    if (content_length > 0) {

        /* leave some room for a trailing '\0' (useful for simple parsing) */

        content = av_malloc(content_length + 1);

        url_read(rt->rtsp_hd, content, content_length);

        content[content_length] = '\0';

    }

    if (content_ptr)

        *content_ptr = content;

}
