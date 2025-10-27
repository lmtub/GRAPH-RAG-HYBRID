static int decode_sequence_header(AVCodecContext *avctx, GetBitContext *gb)

{

    VC9Context *v = avctx->priv_data;



    v->profile = get_bits(gb, 2);

    av_log(avctx, AV_LOG_DEBUG, "Profile: %i\n", v->profile);



#if HAS_ADVANCED_PROFILE

    if (v->profile > PROFILE_MAIN)

    {

        v->level = get_bits(gb, 3);

        v->chromaformat = get_bits(gb, 2);

        if (v->chromaformat != 1)

        {

            av_log(avctx, AV_LOG_ERROR,

                   "Only 4:2:0 chroma format supported\n");

            return -1;

        }

    }

    else

#endif

    {

        v->res_sm = get_bits(gb, 2); //reserved

        if (v->res_sm)

        {

            av_log(avctx, AV_LOG_ERROR,

                   "Reserved RES_SM=%i is forbidden\n", v->res_sm);

            //return -1;

        }

    }



    // (fps-2)/4 (->30)

    v->frmrtq_postproc = get_bits(gb, 3); //common

    // (bitrate-32kbps)/64kbps

    v->bitrtq_postproc = get_bits(gb, 5); //common

    v->loopfilter = get_bits(gb, 1); //common



#if HAS_ADVANCED_PROFILE

    if (v->profile <= PROFILE_MAIN)

#endif

    {

        v->res_x8 = get_bits(gb, 1); //reserved

        if (v->res_x8)

        {

            av_log(avctx, AV_LOG_ERROR,

                   "1 for reserved RES_X8 is forbidden\n");

            return -1;

        }

        v->multires = get_bits(gb, 1);

        v->res_fasttx = get_bits(gb, 1);

        if (!v->res_fasttx)

        {

            av_log(avctx, AV_LOG_ERROR,

                   "0 for reserved RES_FASTTX is forbidden\n");

            //return -1;

        }

    }



    v->fastuvmc =  get_bits(gb, 1); //common

    if (!v->profile && !v->fastuvmc)

    {

        av_log(avctx, AV_LOG_ERROR,

               "FASTUVMC unavailable in Simple Profile\n");

        return -1;

    }

    v->extended_mv =  get_bits(gb, 1); //common

    if (!v->profile && v->extended_mv)

    {

        av_log(avctx, AV_LOG_ERROR,

               "Extended MVs unavailable in Simple Profile\n");

        return -1;

    }

    v->dquant =  get_bits(gb, 2); //common

    v->vstransform =  get_bits(gb, 1); //common

    

#if HAS_ADVANCED_PROFILE

    if (v->profile <= PROFILE_MAIN)

#endif

    {

        v->res_transtab = get_bits(gb, 1);

        if (v->res_transtab)

        {

            av_log(avctx, AV_LOG_ERROR,

                   "1 for reserved RES_TRANSTAB is forbidden\n");

            return -1;

        }

    }



    v->overlap = get_bits(gb, 1); //common



#if HAS_ADVANCED_PROFILE

    if (v->profile <= PROFILE_MAIN)

#endif

    {

        v->syncmarker = get_bits(gb, 1);

        v->rangered = get_bits(gb, 1);

    }



    avctx->max_b_frames = get_bits(gb, 3); //common

    v->quantizer_mode = get_bits(gb, 2); //common



#if HAS_ADVANCED_PROFILE

    if (v->profile <= PROFILE_MAIN)

#endif

    {

        v->finterpflag = get_bits(gb, 1); //common

        v->res_rtm_flag = get_bits(gb, 1); //reserved

        if (!v->res_rtm_flag)

        {

            av_log(avctx, AV_LOG_ERROR,

                   "0 for reserved RES_RTM_FLAG is forbidden\n");

            //return -1;

        }

#if TRACE

        av_log(avctx, AV_LOG_INFO,

               "Profile %i:\nfrmrtq_postproc=%i, bitrtq_postproc=%i\n"

               "LoopFilter=%i, MultiRes=%i, FastUVMV=%i, Extended MV=%i\n"

               "Rangered=%i, VSTransform=%i, Overlap=%i, SyncMarker=%i\n"

               "DQuant=%i, Quantizer mode=%i, Max B frames=%i\n",

               v->profile, v->frmrtq_postproc, v->bitrtq_postproc,

               v->loopfilter, v->multires, v->fastuvmc, v->extended_mv,

               v->rangered, v->vstransform, v->overlap, v->syncmarker,

               v->dquant, v->quantizer_mode, avctx->max_b_frames

               );

#endif

    }

#if HAS_ADVANCED_PROFILE

    else decode_advanced_sequence_header(avctx, gb);

#endif

}
