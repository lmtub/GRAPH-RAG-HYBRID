AVInputFormat *av_probe_input_format3(AVProbeData *pd, int is_opened,

                                      int *score_ret)

{

    AVProbeData lpd = *pd;

    AVInputFormat *fmt1 = NULL, *fmt;

    int score, nodat = 0, score_max = 0;

    const static uint8_t zerobuffer[AVPROBE_PADDING_SIZE];



    if (!lpd.buf)

        lpd.buf = zerobuffer;



    if (lpd.buf_size > 10 && ff_id3v2_match(lpd.buf, ID3v2_DEFAULT_MAGIC)) {

        int id3len = ff_id3v2_tag_len(lpd.buf);

        if (lpd.buf_size > id3len + 16) {

            lpd.buf      += id3len;

            lpd.buf_size -= id3len;

        } else

            nodat = 1;

    }



    fmt = NULL;

    while ((fmt1 = av_iformat_next(fmt1))) {

        if (!is_opened == !(fmt1->flags & AVFMT_NOFILE))

            continue;

        score = 0;

        if (fmt1->read_probe) {

            score = fmt1->read_probe(&lpd);

            if (fmt1->extensions && av_match_ext(lpd.filename, fmt1->extensions))

                score = FFMAX(score, nodat ? AVPROBE_SCORE_EXTENSION / 2 - 1 : 1);

        } else if (fmt1->extensions) {

            if (av_match_ext(lpd.filename, fmt1->extensions))

                score = AVPROBE_SCORE_EXTENSION;

        }

        if (score > score_max) {

            score_max = score;

            fmt       = fmt1;

        } else if (score == score_max)

            fmt = NULL;

    }

    if (nodat)

        score_max = FFMIN(AVPROBE_SCORE_EXTENSION / 2 - 1, score_max);

    *score_ret = score_max;



    return fmt;

}
