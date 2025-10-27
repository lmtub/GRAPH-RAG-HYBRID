int vc1_decode_sequence_header(AVCodecContext *avctx, VC1Context *v, GetBitContext *gb)

{

    av_log(avctx, AV_LOG_DEBUG, "Header: %0X\n", show_bits(gb, 32));

    v->profile = get_bits(gb, 2);

    if (v->profile == PROFILE_COMPLEX)

    {

        av_log(avctx, AV_LOG_WARNING, "WMV3 Complex Profile is not fully supported\n");

    }



    if (v->profile == PROFILE_ADVANCED)

    {

        v->zz_8x4 = ff_vc1_adv_progressive_8x4_zz;

        v->zz_4x8 = ff_vc1_adv_progressive_4x8_zz;

        return decode_sequence_header_adv(v, gb);

    }

    else

    {

        v->zz_8x4 = wmv2_scantableA;

        v->zz_4x8 = wmv2_scantableB;

        v->res_y411   = get_bits1(gb);

        v->res_sprite = get_bits1(gb);

        if (v->res_y411)

        {

            av_log(avctx, AV_LOG_ERROR,

                   "Old interlaced mode is not supported\n");

            return -1;

        }

        if (v->res_sprite) {

            av_log(avctx, AV_LOG_ERROR, "WMVP is not fully supported\n");

        }

    }



    // (fps-2)/4 (->30)

    v->frmrtq_postproc = get_bits(gb, 3); //common

    // (bitrate-32kbps)/64kbps

    v->bitrtq_postproc = get_bits(gb, 5); //common

    v->s.loop_filter = get_bits1(gb); //common

    if(v->s.loop_filter == 1 && v->profile == PROFILE_SIMPLE)

    {

        av_log(avctx, AV_LOG_ERROR,

               "LOOPFILTER shall not be enabled in Simple Profile\n");

    }

    if(v->s.avctx->skip_loop_filter >= AVDISCARD_ALL)

        v->s.loop_filter = 0;



    v->res_x8 = get_bits1(gb); //reserved

    v->multires = get_bits1(gb);

    v->res_fasttx = get_bits1(gb);

    if (!v->res_fasttx)

    {

        v->vc1dsp.vc1_inv_trans_8x8 = ff_simple_idct_8;

        v->vc1dsp.vc1_inv_trans_8x4 = ff_simple_idct84_add;

        v->vc1dsp.vc1_inv_trans_4x8 = ff_simple_idct48_add;

        v->vc1dsp.vc1_inv_trans_4x4 = ff_simple_idct44_add;

        v->vc1dsp.vc1_inv_trans_8x8_dc = ff_simple_idct_add_8;

        v->vc1dsp.vc1_inv_trans_8x4_dc = ff_simple_idct84_add;

        v->vc1dsp.vc1_inv_trans_4x8_dc = ff_simple_idct48_add;

        v->vc1dsp.vc1_inv_trans_4x4_dc = ff_simple_idct44_add;

    }



    v->fastuvmc =  get_bits1(gb); //common

    if (!v->profile && !v->fastuvmc)

    {

        av_log(avctx, AV_LOG_ERROR,

               "FASTUVMC unavailable in Simple Profile\n");

        return -1;

    }

    v->extended_mv =  get_bits1(gb); //common

    if (!v->profile && v->extended_mv)

    {

        av_log(avctx, AV_LOG_ERROR,

               "Extended MVs unavailable in Simple Profile\n");

        return -1;

    }

    v->dquant =  get_bits(gb, 2); //common

    v->vstransform =  get_bits1(gb); //common



    v->res_transtab = get_bits1(gb);

    if (v->res_transtab)

    {

        av_log(avctx, AV_LOG_ERROR,

               "1 for reserved RES_TRANSTAB is forbidden\n");

        return -1;

    }



    v->overlap = get_bits1(gb); //common



    v->s.resync_marker = get_bits1(gb);

    v->rangered = get_bits1(gb);

    if (v->rangered && v->profile == PROFILE_SIMPLE)

    {

        av_log(avctx, AV_LOG_INFO,

               "RANGERED should be set to 0 in Simple Profile\n");

    }



    v->s.max_b_frames = avctx->max_b_frames = get_bits(gb, 3); //common

    v->quantizer_mode = get_bits(gb, 2); //common



    v->finterpflag = get_bits1(gb); //common



    if (v->res_sprite) {

        v->s.avctx->width  = v->s.avctx->coded_width  = get_bits(gb, 11);

        v->s.avctx->height = v->s.avctx->coded_height = get_bits(gb, 11);

        skip_bits(gb, 5); //frame rate

        v->res_x8 = get_bits1(gb);

        if (get_bits1(gb)) { // something to do with DC VLC selection

            av_log(avctx, AV_LOG_ERROR, "Unsupported sprite feature\n");

            return -1;

        }

        skip_bits(gb, 3); //slice code

        v->res_rtm_flag = 0;

    } else {

        v->res_rtm_flag = get_bits1(gb); //reserved

    }

    if (!v->res_rtm_flag)

    {

//            av_log(avctx, AV_LOG_ERROR,

//                   "0 for reserved RES_RTM_FLAG is forbidden\n");

        av_log(avctx, AV_LOG_ERROR,

               "Old WMV3 version detected, some frames may be decoded incorrectly\n");

        //return -1;

    }

    //TODO: figure out what they mean (always 0x402F)

    if(!v->res_fasttx) skip_bits(gb, 16);

    av_log(avctx, AV_LOG_DEBUG,

               "Profile %i:\nfrmrtq_postproc=%i, bitrtq_postproc=%i\n"

               "LoopFilter=%i, MultiRes=%i, FastUVMC=%i, Extended MV=%i\n"

               "Rangered=%i, VSTransform=%i, Overlap=%i, SyncMarker=%i\n"

               "DQuant=%i, Quantizer mode=%i, Max B frames=%i\n",

               v->profile, v->frmrtq_postproc, v->bitrtq_postproc,

               v->s.loop_filter, v->multires, v->fastuvmc, v->extended_mv,

               v->rangered, v->vstransform, v->overlap, v->s.resync_marker,

               v->dquant, v->quantizer_mode, avctx->max_b_frames

               );

    return 0;

}
