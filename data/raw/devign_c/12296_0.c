static int64_t mp3_sync(AVFormatContext *s, int64_t target_pos, int flags)

{

    int dir = (flags&AVSEEK_FLAG_BACKWARD) ? -1 : 1;

    int64_t best_pos;

    int best_score, i, j;

    int64_t ret;



    avio_seek(s->pb, FFMAX(target_pos - SEEK_WINDOW, 0), SEEK_SET);

    ret = avio_seek(s->pb, target_pos, SEEK_SET);

    if (ret < 0)

        return ret;



#define MIN_VALID 3

    best_pos = target_pos;

    best_score = 999;

    for(i=0; i<SEEK_WINDOW; i++) {

        int64_t pos = target_pos + (dir > 0 ? i - SEEK_WINDOW/4 : -i);

        int64_t candidate = -1;

        int score = 999;



        if (pos < 0)

            continue;



        for(j=0; j<MIN_VALID; j++) {

            ret = check(s->pb, pos);

            if(ret < 0)

                break;

            if ((target_pos - pos)*dir <= 0 && abs(MIN_VALID/2-j) < score) {

                candidate = pos;

                score = abs(MIN_VALID/2-j);

            }

            pos += ret;

        }

        if (best_score > score && j == MIN_VALID) {

            best_pos = candidate;

            best_score = score;

            if(score == 0)

                break;

        }

    }



    return avio_seek(s->pb, best_pos, SEEK_SET);

}
