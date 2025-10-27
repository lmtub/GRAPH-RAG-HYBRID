static int decode_value(SCPRContext *s, unsigned *cnt, unsigned maxc, unsigned step, unsigned *rval)

{

    GetByteContext *gb = &s->gb;

    RangeCoder *rc = &s->rc;

    unsigned totfr = cnt[maxc];

    unsigned value;

    unsigned c = 0, cumfr = 0, cnt_c = 0;

    int i, ret;



    if ((ret = s->get_freq(rc, totfr, &value)) < 0)

        return ret;



    while (c < maxc) {

        cnt_c = cnt[c];

        if (value >= cumfr + cnt_c)

            cumfr += cnt_c;

        else

            break;

        c++;

    }

    s->decode(gb, rc, cumfr, cnt_c, totfr);



    cnt[c] = cnt_c + step;

    totfr += step;

    if (totfr > BOT) {

        totfr = 0;

        for (i = 0; i < maxc; i++) {

            unsigned nc = (cnt[i] >> 1) + 1;

            cnt[i] = nc;

            totfr += nc;

        }

    }



    cnt[maxc] = totfr;

    *rval = c;



    return 0;

}
