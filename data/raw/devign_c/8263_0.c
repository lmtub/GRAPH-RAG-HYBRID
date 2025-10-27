void avcodec_get_context_defaults(AVCodecContext *s){

    memset(s, 0, sizeof(AVCodecContext));



    s->av_class= &av_codec_context_class;

    s->bit_rate= 800*1000;

    s->bit_rate_tolerance= s->bit_rate*10;

    s->qmin= 2;

    s->qmax= 31;

    s->mb_qmin= 2;

    s->mb_qmax= 31;

    s->rc_eq= "tex^qComp";

    s->qcompress= 0.5;

    s->max_qdiff= 3;

    s->b_quant_factor=1.25;

    s->b_quant_offset=1.25;

    s->i_quant_factor=-0.8;

    s->i_quant_offset=0.0;

    s->error_concealment= 3;

    s->error_resilience= 1;

    s->workaround_bugs= FF_BUG_AUTODETECT;

    s->frame_rate_base= 1;

    s->frame_rate = 25;

    s->gop_size= 50;

    s->me_method= ME_EPZS;

    s->get_buffer= avcodec_default_get_buffer;

    s->release_buffer= avcodec_default_release_buffer;

    s->get_format= avcodec_default_get_format;

    s->execute= avcodec_default_execute;

    s->thread_count=1;

    s->me_subpel_quality=8;

    s->lmin= FF_QP2LAMBDA * s->qmin;

    s->lmax= FF_QP2LAMBDA * s->qmax;

    s->sample_aspect_ratio= (AVRational){0,1};

    s->ildct_cmp= FF_CMP_VSAD;

    s->profile= FF_PROFILE_UNKNOWN;

    s->level= FF_LEVEL_UNKNOWN;

    

    s->intra_quant_bias= FF_DEFAULT_QUANT_BIAS;

    s->inter_quant_bias= FF_DEFAULT_QUANT_BIAS;

    s->palctrl = NULL;

    s->reget_buffer= avcodec_default_reget_buffer;

}
