static int mov_probe(AVProbeData *p)

{

    int64_t offset;

    uint32_t tag;

    int score = 0;



    /* check file header */

    offset = 0;

    for (;;) {

        /* ignore invalid offset */

        if ((offset + 8) > (unsigned int)p->buf_size)

            return score;

        tag = AV_RL32(p->buf + offset + 4);

        switch(tag) {

        /* check for obvious tags */

        case MKTAG('j','P',' ',' '): /* jpeg 2000 signature */

        case MKTAG('m','o','o','v'):

        case MKTAG('m','d','a','t'):

        case MKTAG('p','n','o','t'): /* detect movs with preview pics like ew.mov and april.mov */

        case MKTAG('u','d','t','a'): /* Packet Video PVAuthor adds this and a lot of more junk */

        case MKTAG('f','t','y','p'):

            return AVPROBE_SCORE_MAX;

        /* those are more common words, so rate then a bit less */

        case MKTAG('e','d','i','w'): /* xdcam files have reverted first tags */

        case MKTAG('w','i','d','e'):

        case MKTAG('f','r','e','e'):

        case MKTAG('j','u','n','k'):

        case MKTAG('p','i','c','t'):

            return AVPROBE_SCORE_MAX - 5;

        case MKTAG(0x82,0x82,0x7f,0x7d):

        case MKTAG('s','k','i','p'):

        case MKTAG('u','u','i','d'):

        case MKTAG('p','r','f','l'):

            offset = AV_RB32(p->buf+offset) + offset;

            /* if we only find those cause probedata is too small at least rate them */

            score = AVPROBE_SCORE_MAX - 50;

            break;

        default:

            /* unrecognized tag */

            return score;

        }

    }

}
