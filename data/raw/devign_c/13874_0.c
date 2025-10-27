AVInputFormat *av_probe_input_format2(AVProbeData *pd, int is_opened, int *score_max)

{

    AVProbeData lpd = *pd;

    AVInputFormat *fmt1 = NULL, *fmt;

    int score, id3 = 0;



    if (lpd.buf_size > 10 && ff_id3v2_match(lpd.buf, ID3v2_DEFAULT_MAGIC)) {

        int id3len = ff_id3v2_tag_len(lpd.buf);

        if (lpd.buf_size > id3len + 16) {

            lpd.buf += id3len;

            lpd.buf_size -= id3len;

        }

        id3 = 1;

    }



    fmt = NULL;

    while ((fmt1 = av_iformat_next(fmt1))) {

        if (!is_opened == !(fmt1->flags & AVFMT_NOFILE))

            continue;

        score = 0;

        if (fmt1->read_probe) {

            score = fmt1->read_probe(&lpd);

        } else if (fmt1->extensions) {

            if (av_match_ext(lpd.filename, fmt1->extensions)) {

                score = 50;

            }

        }

        if (score > *score_max) {

            *score_max = score;

            fmt = fmt1;

        }else if (score == *score_max)

            fmt = NULL;

    }



    /* a hack for files with huge id3v2 tags -- try to guess by file extension. */

    if (!fmt && id3 && *score_max < AVPROBE_SCORE_MAX/4) {

        while ((fmt = av_iformat_next(fmt)))

            if (fmt->extensions && av_match_ext(lpd.filename, fmt->extensions)) {

                *score_max = AVPROBE_SCORE_MAX/4;

                break;

            }

    }



    return fmt;

}
