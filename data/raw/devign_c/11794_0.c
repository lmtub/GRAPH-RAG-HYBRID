static void filter_line_c(uint8_t *dst,

                          uint8_t *prev, uint8_t *cur, uint8_t *next,

                          int w, int prefs, int mrefs, int parity, int mode)

{

    int x;

    uint8_t *prev2 = parity ? prev : cur ;

    uint8_t *next2 = parity ? cur  : next;

    for (x = 0;  x < w; x++) {

        int c = cur[mrefs];

        int d = (prev2[0] + next2[0])>>1;

        int e = cur[prefs];

        int temporal_diff0 = FFABS(prev2[0] - next2[0]);

        int temporal_diff1 =(FFABS(prev[mrefs] - c) + FFABS(prev[prefs] - e) )>>1;

        int temporal_diff2 =(FFABS(next[mrefs] - c) + FFABS(next[prefs] - e) )>>1;

        int diff = FFMAX3(temporal_diff0>>1, temporal_diff1, temporal_diff2);

        int spatial_pred = (c+e)>>1;

        int spatial_score = FFABS(cur[mrefs-1] - cur[prefs-1]) + FFABS(c-e)

                          + FFABS(cur[mrefs+1] - cur[prefs+1]) - 1;



#define CHECK(j)\

    {   int score = FFABS(cur[mrefs-1+j] - cur[prefs-1-j])\

                  + FFABS(cur[mrefs  +j] - cur[prefs  -j])\

                  + FFABS(cur[mrefs+1+j] - cur[prefs+1-j]);\

        if (score < spatial_score) {\

            spatial_score= score;\

            spatial_pred= (cur[mrefs  +j] + cur[prefs  -j])>>1;\



        CHECK(-1) CHECK(-2) }} }}
