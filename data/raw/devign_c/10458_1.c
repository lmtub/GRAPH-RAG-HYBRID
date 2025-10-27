static int64_t calculate_mode_score(CinepakEncContext *s, CinepakMode mode, int h, int v1_size, int v4_size, int v4, strip_info *info)

{

    //score = FF_LAMBDA_SCALE * error + lambda * bits

    int x;

    int entry_size = s->pix_fmt == AV_PIX_FMT_YUV420P ? 6 : 4;

    int mb_count = s->w * h / MB_AREA;

    mb_info *mb;

    int64_t score1, score2, score3;

    int64_t ret = s->lambda * ((v1_size ? CHUNK_HEADER_SIZE + v1_size * entry_size : 0) +

                   (v4_size ? CHUNK_HEADER_SIZE + v4_size * entry_size : 0) +

                   CHUNK_HEADER_SIZE) << 3;



    //av_log(s->avctx, AV_LOG_INFO, "sizes %3i %3i -> %9li score mb_count %i", v1_size, v4_size, ret, mb_count);



    switch(mode) {

    case MODE_V1_ONLY:

        //one byte per MB

        ret += s->lambda * 8 * mb_count;



        for(x = 0; x < mb_count; x++) {

            mb = &s->mb[x];

            ret += FF_LAMBDA_SCALE * mb->v1_error;

            mb->best_encoding = ENC_V1;

        }



        break;

    case MODE_V1_V4:

        //9 or 33 bits per MB

        for(x = 0; x < mb_count; x++) {

            mb = &s->mb[x];

            score1 = s->lambda * 9  + FF_LAMBDA_SCALE * mb->v1_error;

            score2 = s->lambda * 33 + FF_LAMBDA_SCALE * mb->v4_error[v4];



            if(score1 <= score2) {

                ret += score1;

                mb->best_encoding = ENC_V1;

            } else {

                ret += score2;

                mb->best_encoding = ENC_V4;

            }

        }



        break;

    case MODE_MC:

        //1, 10 or 34 bits per MB

        for(x = 0; x < mb_count; x++) {

            mb = &s->mb[x];

            score1 = s->lambda * 1  + FF_LAMBDA_SCALE * mb->skip_error;

            score2 = s->lambda * 10 + FF_LAMBDA_SCALE * mb->v1_error;

            score3 = s->lambda * 34 + FF_LAMBDA_SCALE * mb->v4_error[v4];





            if(score1 <= score2 && score1 <= score3) {

                ret += score1;

                mb->best_encoding = ENC_SKIP;

            } else if(score2 <= score1 && score2 <= score3) {

                ret += score2;

                mb->best_encoding = ENC_V1;

            } else {

                ret += score3;

                mb->best_encoding = ENC_V4;

            }

        }



        break;

    }



    return ret;

}
