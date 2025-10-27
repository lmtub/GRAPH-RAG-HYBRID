static int sdp_probe(AVProbeData *p1)

{

    const char *p = p1->buf, *p_end = p1->buf + p1->buf_size;



    /* we look for a line beginning "c=IN IP4" */

    while (p < p_end && *p != '\0') {

        if (p + sizeof("c=IN IP4") - 1 < p_end && av_strstart(p, "c=IN IP4", NULL))

            return AVPROBE_SCORE_MAX / 2;



        while(p < p_end - 1 && *p != '\n') p++;

        if (++p >= p_end)

            break;

        if (*p == '\r')

            p++;

    }

    return 0;

}
