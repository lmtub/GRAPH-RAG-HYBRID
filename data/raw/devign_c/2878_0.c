static int mpegts_probe(AVProbeData *p)

{

    const int size = p->buf_size;

    int maxscore = 0;

    int sumscore = 0;

    int i;

    int check_count = size / TS_FEC_PACKET_SIZE;

#define CHECK_COUNT 10

#define CHECK_BLOCK 100



    if (check_count < CHECK_COUNT)

        return 0;



    for (i = 0; i<check_count; i+=CHECK_BLOCK) {

        int left = FFMIN(check_count - i, CHECK_BLOCK);

        int score      = analyze(p->buf + TS_PACKET_SIZE     *i, TS_PACKET_SIZE     *left, TS_PACKET_SIZE     , NULL, 1);

        int dvhs_score = analyze(p->buf + TS_DVHS_PACKET_SIZE*i, TS_DVHS_PACKET_SIZE*left, TS_DVHS_PACKET_SIZE, NULL, 1);

        int fec_score  = analyze(p->buf + TS_FEC_PACKET_SIZE *i, TS_FEC_PACKET_SIZE *left, TS_FEC_PACKET_SIZE , NULL, 1);

        score = FFMAX3(score, dvhs_score, fec_score);

        sumscore += score;

        maxscore = FFMAX(maxscore, score);

    }



    sumscore = sumscore * CHECK_COUNT / check_count;

    maxscore = maxscore * CHECK_COUNT / CHECK_BLOCK;



    av_dlog(0, "TS score: %d %d\n", sumscore, maxscore);



    if      (sumscore > 6) return AVPROBE_SCORE_MAX   + sumscore - CHECK_COUNT;

    else if (maxscore > 6) return AVPROBE_SCORE_MAX/2 + sumscore - CHECK_COUNT;

    else

        return 0;

}
