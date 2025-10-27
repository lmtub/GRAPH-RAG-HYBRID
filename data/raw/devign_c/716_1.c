static int rd_frame(CinepakEncContext *s, AVFrame *frame, unsigned char *buf, int buf_size)

{

    int num_strips, strip, h, i, y, size, temp_size, best_size;

    AVPicture last_pict, pict, scratch_pict;

    int64_t best_score = 0, score, score_temp;



    //TODO: support encoding zero strips (meaning skip the whole frame)

    for(num_strips = MIN_STRIPS; num_strips <= MAX_STRIPS && num_strips <= s->h / MB_SIZE; num_strips++) {

        score = 0;

        size = 0;

        h = s->h / num_strips;

        //make h into next multiple of 4

        h += 4 - (h & 3);



        for(strip = 0; strip < num_strips; strip++) {

            y = strip*h;



            get_sub_picture(s, 0, y, (AVPicture*)frame,            &pict);

            get_sub_picture(s, 0, y, (AVPicture*)&s->last_frame,    &last_pict);

            get_sub_picture(s, 0, y, (AVPicture*)&s->scratch_frame, &scratch_pict);



            if((temp_size = rd_strip(s, y, FFMIN(h, s->h - y), frame->key_frame, &last_pict, &pict, &scratch_pict, s->frame_buf + CVID_HEADER_SIZE, &score_temp)) < 0)

                return temp_size;



            score += score_temp;

            size += temp_size;

        }



        if(best_score == 0 || score < best_score) {

            best_score = score;

            best_size = size + write_cvid_header(s, s->frame_buf, num_strips, size);

            av_log(s->avctx, AV_LOG_INFO, "best number of strips so far: %2i, %12li, %i B\n", num_strips, score, best_size);



            FFSWAP(AVFrame, s->best_frame, s->scratch_frame);

        }

    }



    memcpy(buf, s->frame_buf, best_size);



    return best_size;

}
