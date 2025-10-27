static int rd_frame(CinepakEncContext *s, const AVFrame *frame, int isakeyframe, unsigned char *buf, int buf_size)

{

    int num_strips, strip, i, y, nexty, size, temp_size, best_size;

    AVPicture last_pict, pict, scratch_pict;

    int64_t best_score = 0, score, score_temp;

#ifdef CINEPAK_REPORT_SERR

    int64_t best_serr = 0, serr, serr_temp;

#endif



    int best_nstrips;



    if(s->pix_fmt == AV_PIX_FMT_RGB24) {

        int x;

// build a copy of the given frame in the correct colorspace

        for(y = 0; y < s->h; y += 2) {

            for(x = 0; x < s->w; x += 2) {

                uint8_t *ir[2]; int32_t r, g, b, rr, gg, bb;

                ir[0] = ((AVPicture*)frame)->data[0] + x*3 + y*((AVPicture*)frame)->linesize[0];

                ir[1] = ir[0] + ((AVPicture*)frame)->linesize[0];

                get_sub_picture(s, x, y, (AVPicture*)s->input_frame, &scratch_pict);

                r = g = b = 0;

                for(i=0; i<4; ++i) {

                    int i1, i2;

                    i1 = (i&1); i2 = (i>=2);

                    rr = ir[i2][i1*3+0];

                    gg = ir[i2][i1*3+1];

                    bb = ir[i2][i1*3+2];

                    r += rr; g += gg; b += bb;

// using fixed point arithmetic for portable repeatability, scaling by 2^23

// "Y"

//                    rr = 0.2857*rr + 0.5714*gg + 0.1429*bb;

                    rr = (2396625*rr + 4793251*gg + 1198732*bb) >> 23;

                    if(      rr <   0) rr =   0;

                    else if (rr > 255) rr = 255;

                    scratch_pict.data[0][i1 + i2*scratch_pict.linesize[0]] = rr;

                }

// let us scale down as late as possible

//                r /= 4; g /= 4; b /= 4;

// "U"

//                rr = -0.1429*r - 0.2857*g + 0.4286*b;

                rr = (-299683*r - 599156*g + 898839*b) >> 23;

                if(      rr < -128) rr = -128;

                else if (rr >  127) rr =  127;

                scratch_pict.data[1][0] = rr + 128; // quantize needs unsigned

// "V"

//                rr = 0.3571*r - 0.2857*g - 0.0714*b;

                rr = (748893*r - 599156*g - 149737*b) >> 23;

                if(      rr < -128) rr = -128;

                else if (rr >  127) rr =  127;

                scratch_pict.data[2][0] = rr + 128; // quantize needs unsigned

            }

        }

    }



    //would be nice but quite certainly incompatible with vintage players:

    // support encoding zero strips (meaning skip the whole frame)

    for(num_strips = s->min_strips; num_strips <= s->max_strips && num_strips <= s->h / MB_SIZE; num_strips++) {

        score = 0;

        size = 0;

#ifdef CINEPAK_REPORT_SERR

        serr = 0;

#endif



        for(y = 0, strip = 1; y < s->h; strip++, y = nexty) {

            int strip_height;



            nexty = strip * s->h / num_strips; // <= s->h

            //make nexty the next multiple of 4 if not already there

            if(nexty & 3)

                nexty += 4 - (nexty & 3);



            strip_height = nexty - y;

            if(strip_height <= 0) { // can this ever happen?

                av_log(s->avctx, AV_LOG_INFO, "skipping zero height strip %i of %i\n", strip, num_strips);

                continue;

            }



            if(s->pix_fmt == AV_PIX_FMT_RGB24)

                get_sub_picture(s, 0, y, (AVPicture*)s->input_frame,    &pict);

            else

                get_sub_picture(s, 0, y, (AVPicture*)frame,              &pict);

            get_sub_picture(s, 0, y, (AVPicture*)s->last_frame,    &last_pict);

            get_sub_picture(s, 0, y, (AVPicture*)s->scratch_frame, &scratch_pict);



            if((temp_size = rd_strip(s, y, strip_height, isakeyframe, &last_pict, &pict, &scratch_pict, s->frame_buf + size + CVID_HEADER_SIZE, &score_temp

#ifdef CINEPAK_REPORT_SERR

, &serr_temp

#endif

)) < 0)

                return temp_size;



            score += score_temp;

#ifdef CINEPAK_REPORT_SERR

            serr += serr_temp;

#endif

            size += temp_size;

            //av_log(s->avctx, AV_LOG_INFO, "strip %d, isakeyframe=%d", strip, isakeyframe);

            //av_log(s->avctx, AV_LOG_INFO, "\n");

        }



        if(best_score == 0 || score < best_score) {

            best_score = score;

#ifdef CINEPAK_REPORT_SERR

            best_serr = serr;

#endif

            best_size = size + write_cvid_header(s, s->frame_buf, num_strips, size, isakeyframe);

            //av_log(s->avctx, AV_LOG_INFO, "best number of strips so far: %2i, %12lli, %i B\n", num_strips, (long long int)score, best_size);

#ifdef CINEPAK_REPORT_SERR

            av_log(s->avctx, AV_LOG_INFO, "best number of strips so far: %2i, %12lli, %i B\n", num_strips, (long long int)serr, best_size);

#endif



            FFSWAP(AVFrame *, s->best_frame, s->scratch_frame);

            memcpy(buf, s->frame_buf, best_size);

            best_nstrips = num_strips;

        }

// avoid trying too many strip numbers without a real reason

// (this makes the processing of the very first frame faster)

        if(num_strips - best_nstrips > 4)

            break;

    }



// let the number of strips slowly adapt to the changes in the contents,

// compared to full bruteforcing every time this will occasionally lead

// to some r/d performance loss but makes encoding up to several times faster

    if(!s->strip_number_delta_range) {

        if(best_nstrips == s->max_strips) { // let us try to step up

            s->max_strips = best_nstrips + 1;

            if(s->max_strips >= s->max_max_strips)

                s->max_strips = s->max_max_strips;

        } else { // try to step down

            s->max_strips = best_nstrips;

        }

        s->min_strips = s->max_strips - 1;

        if(s->min_strips < s->min_min_strips)

            s->min_strips = s->min_min_strips;

    } else {

        s->max_strips = best_nstrips + s->strip_number_delta_range;

        if(s->max_strips >= s->max_max_strips)

            s->max_strips = s->max_max_strips;

        s->min_strips = best_nstrips - s->strip_number_delta_range;

        if(s->min_strips < s->min_min_strips)

            s->min_strips = s->min_min_strips;

    }



    return best_size;

}
