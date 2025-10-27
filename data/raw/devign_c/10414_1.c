static int decode_sequence_header_adv(VC1Context *v, GetBitContext *gb)

{

    v->res_rtm_flag = 1;

    v->level = get_bits(gb, 3);

    if(v->level >= 5)

    {

        av_log(v->s.avctx, AV_LOG_ERROR, "Reserved LEVEL %i\n",v->level);

    }

    v->chromaformat = get_bits(gb, 2);

    if (v->chromaformat != 1)

    {

        av_log(v->s.avctx, AV_LOG_ERROR,

               "Only 4:2:0 chroma format supported\n");

        return -1;

    }



    // (fps-2)/4 (->30)

    v->frmrtq_postproc = get_bits(gb, 3); //common

    // (bitrate-32kbps)/64kbps

    v->bitrtq_postproc = get_bits(gb, 5); //common

    v->postprocflag = get_bits(gb, 1); //common



    v->s.avctx->coded_width = (get_bits(gb, 12) + 1) << 1;

    v->s.avctx->coded_height = (get_bits(gb, 12) + 1) << 1;

    v->broadcast = get_bits1(gb);

    v->interlace = get_bits1(gb);

    if(v->interlace){

        av_log(v->s.avctx, AV_LOG_ERROR, "Interlaced mode not supported (yet)\n");

        return -1;

    }

    v->tfcntrflag = get_bits1(gb);

    v->finterpflag = get_bits1(gb);

    get_bits1(gb); // reserved

    v->psf = get_bits1(gb);

    if(v->psf) { //PsF, 6.1.13

        av_log(v->s.avctx, AV_LOG_ERROR, "Progressive Segmented Frame mode: not supported (yet)\n");

        return -1;

    }

    if(get_bits1(gb)) { //Display Info - decoding is not affected by it

        int w, h, ar = 0;

        av_log(v->s.avctx, AV_LOG_INFO, "Display extended info:\n");

        w = get_bits(gb, 14);

        h = get_bits(gb, 14);

        av_log(v->s.avctx, AV_LOG_INFO, "Display dimensions: %ix%i\n", w, h);

        //TODO: store aspect ratio in AVCodecContext

        if(get_bits1(gb))

            ar = get_bits(gb, 4);

        if(ar == 15) {

            w = get_bits(gb, 8);

            h = get_bits(gb, 8);

        }



        if(get_bits1(gb)){ //framerate stuff

            if(get_bits1(gb)) {

                get_bits(gb, 16);

            } else {

                get_bits(gb, 8);

                get_bits(gb, 4);

            }

        }



        if(get_bits1(gb)){

            v->color_prim = get_bits(gb, 8);

            v->transfer_char = get_bits(gb, 8);

            v->matrix_coef = get_bits(gb, 8);

        }

    }



    v->hrd_param_flag = get_bits1(gb);

    if(v->hrd_param_flag) {

        int i;

        v->hrd_num_leaky_buckets = get_bits(gb, 5);

        get_bits(gb, 4); //bitrate exponent

        get_bits(gb, 4); //buffer size exponent

        for(i = 0; i < v->hrd_num_leaky_buckets; i++) {

            get_bits(gb, 16); //hrd_rate[n]

            get_bits(gb, 16); //hrd_buffer[n]

        }

    }

    return 0;

}
