int mpeg4_decode_video_packet_header(MpegEncContext *s)

{

    int mb_num_bits= av_log2(s->mb_num - 1) + 1;

    int header_extension=0, mb_num, len;



    /* is there enough space left for a video packet + header */

    if( get_bits_count(&s->gb) > s->gb.size_in_bits-20) return -1;



    for(len=0; len<32; len++){

        if(get_bits1(&s->gb)) break;

    }



    if(len!=ff_mpeg4_get_video_packet_prefix_length(s)){

        av_log(s->avctx, AV_LOG_ERROR, "marker does not match f_code\n");

        return -1;

    }



    if(s->shape != RECT_SHAPE){

        header_extension= get_bits1(&s->gb);

        //FIXME more stuff here

    }



    mb_num= get_bits(&s->gb, mb_num_bits);

    if(mb_num>=s->mb_num){

        av_log(s->avctx, AV_LOG_ERROR, "illegal mb_num in video packet (%d %d) \n", mb_num, s->mb_num);

        return -1;

    }



    s->mb_x= mb_num % s->mb_width;

    s->mb_y= mb_num / s->mb_width;



    if(s->shape != BIN_ONLY_SHAPE){

        int qscale= get_bits(&s->gb, s->quant_precision);

        if(qscale)

            s->chroma_qscale=s->qscale= qscale;

    }



    if(s->shape == RECT_SHAPE){

        header_extension= get_bits1(&s->gb);

    }

    if(header_extension){

        int time_incr=0;



        while (get_bits1(&s->gb) != 0)

            time_incr++;



        check_marker(&s->gb, "before time_increment in video packed header");

        skip_bits(&s->gb, s->time_increment_bits); /* time_increment */

        check_marker(&s->gb, "before vop_coding_type in video packed header");



        skip_bits(&s->gb, 2); /* vop coding type */

        //FIXME not rect stuff here



        if(s->shape != BIN_ONLY_SHAPE){

            skip_bits(&s->gb, 3); /* intra dc vlc threshold */

//FIXME don't just ignore everything

            if(s->pict_type == AV_PICTURE_TYPE_S && s->vol_sprite_usage==GMC_SPRITE){

                mpeg4_decode_sprite_trajectory(s, &s->gb);

                av_log(s->avctx, AV_LOG_ERROR, "untested\n");

            }



            //FIXME reduced res stuff here



            if (s->pict_type != AV_PICTURE_TYPE_I) {

                int f_code = get_bits(&s->gb, 3);       /* fcode_for */

                if(f_code==0){

                    av_log(s->avctx, AV_LOG_ERROR, "Error, video packet header damaged (f_code=0)\n");

                }

            }

            if (s->pict_type == AV_PICTURE_TYPE_B) {

                int b_code = get_bits(&s->gb, 3);

                if(b_code==0){

                    av_log(s->avctx, AV_LOG_ERROR, "Error, video packet header damaged (b_code=0)\n");

                }

            }

        }

    }

    //FIXME new-pred stuff



    return 0;

}
