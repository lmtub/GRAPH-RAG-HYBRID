static AVInputFormat *probe_input_format(AVProbeData *pd, int is_opened)

{

    AVInputFormat *fmt1, *fmt;

    int score, score_max;



    fmt = NULL;

    score_max = 0;

    for(fmt1 = first_iformat; fmt1 != NULL; fmt1 = fmt1->next) {

        if (!is_opened && !(fmt1->flags & AVFMT_NOFILE))

            continue;

        score = 0;

        if (fmt1->extensions) {

            if (match_ext(pd->filename, fmt1->extensions)) {

                score = 50;

            }

        } else if (fmt1->read_probe) {

            score = fmt1->read_probe(pd);

        }

        if (score > score_max) {

            score_max = score;

            fmt = fmt1;

        }

    }

    return fmt;

}
