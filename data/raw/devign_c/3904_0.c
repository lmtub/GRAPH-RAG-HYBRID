static int hls_probe(AVProbeData *p)

{

    /* Require #EXTM3U at the start, and either one of the ones below

     * somewhere for a proper match. */

    if (strncmp(p->buf, "#EXTM3U", 7))

        return 0;



    if (p->filename && !av_match_ext(p->filename, "m3u8,m3u"))

        return 0;



    if (strstr(p->buf, "#EXT-X-STREAM-INF:")     ||

        strstr(p->buf, "#EXT-X-TARGETDURATION:") ||

        strstr(p->buf, "#EXT-X-MEDIA-SEQUENCE:"))

        return AVPROBE_SCORE_MAX;

    return 0;

}
