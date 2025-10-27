static int nsv_probe(AVProbeData *p)

{

    int i;

    av_dlog(NULL, "nsv_probe(), buf_size %d\n", p->buf_size);

    /* check file header */

    /* streamed files might not have any header */

    if (p->buf[0] == 'N' && p->buf[1] == 'S' &&

        p->buf[2] == 'V' && (p->buf[3] == 'f' || p->buf[3] == 's'))

        return AVPROBE_SCORE_MAX;

    /* XXX: do streamed files always start at chunk boundary ?? */

    /* or do we need to search NSVs in the byte stream ? */

    /* seems the servers don't bother starting clean chunks... */

    /* sometimes even the first header is at 9KB or something :^) */

    for (i = 1; i < p->buf_size - 3; i++) {

        if (p->buf[i+0] == 'N' && p->buf[i+1] == 'S' &&

            p->buf[i+2] == 'V' && p->buf[i+3] == 's')

            return AVPROBE_SCORE_MAX-20;

    }

    /* so we'll have more luck on extension... */

    if (av_match_ext(p->filename, "nsv"))

        return AVPROBE_SCORE_MAX/2;

    /* FIXME: add mime-type check */

    return 0;

}
