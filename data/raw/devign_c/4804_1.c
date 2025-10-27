AVRational ff_choose_timebase(AVFormatContext *s, AVStream *st, int min_precission)

{

    AVRational q;

    int j;



    if (st->codec->codec_type == AVMEDIA_TYPE_AUDIO) {

        q = (AVRational){1, st->codec->sample_rate};

    } else {

        q = st->codec->time_base;

    }

    for (j=2; j<2000; j+= 1+(j>2))

        while (q.den / q.num < min_precission && q.num % j == 0)

            q.num /= j;

    while (q.den / q.num < min_precission && q.den < (1<<24))

        q.den <<= 1;



    return q;

}
