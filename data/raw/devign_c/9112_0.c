static int http_connect(URLContext *h, const char *path, const char *hoststr)

{

    HTTPContext *s = h->priv_data;

    int post, err, ch;

    char line[1024], *q;





    /* send http header */

    post = h->flags & URL_WRONLY;



    snprintf(s->buffer, sizeof(s->buffer),

             "%s %s HTTP/1.0\r\n"

             "User-Agent: %s\r\n"

             "Accept: */*\r\n"

             "Host: %s\r\n"

             "\r\n",

             post ? "POST" : "GET",

             path,

             LIBAVFORMAT_IDENT,

             hoststr);

    

    if (http_write(h, s->buffer, strlen(s->buffer)) < 0)

        return AVERROR_IO;

        

    /* init input buffer */

    s->buf_ptr = s->buffer;

    s->buf_end = s->buffer;

    s->line_count = 0;

    s->location[0] = '\0';

    if (post) {

        sleep(1);

        return 0;

    }

    

    /* wait for header */

    q = line;

    for(;;) {

        ch = http_getc(s);

        if (ch < 0)

            return AVERROR_IO;

        if (ch == '\n') {

            /* process line */

            if (q > line && q[-1] == '\r')

                q--;

            *q = '\0';

#ifdef DEBUG

            printf("header='%s'\n", line);

#endif

            err = process_line(s, line, s->line_count);

            if (err < 0)

                return err;

            if (err == 0)

                return 0;

            s->line_count++;

            q = line;

        } else {

            if ((q - line) < sizeof(line) - 1)

                *q++ = ch;

        }

    }

}
