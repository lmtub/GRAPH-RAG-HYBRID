static int estimate_qp(MpegEncContext *s, int dry_run){

    if (s->next_lambda){

        s->current_picture_ptr->f.quality =

        s->current_picture.f.quality = s->next_lambda;

        if(!dry_run) s->next_lambda= 0;

    } else if (!s->fixed_qscale) {

        s->current_picture_ptr->f.quality =

        s->current_picture.f.quality = ff_rate_estimate_qscale(s, dry_run);

        if (s->current_picture.f.quality < 0)

            return -1;

    }



    if(s->adaptive_quant){

        switch(s->codec_id){

        case AV_CODEC_ID_MPEG4:

            if (CONFIG_MPEG4_ENCODER)

                ff_clean_mpeg4_qscales(s);

            break;

        case AV_CODEC_ID_H263:

        case AV_CODEC_ID_H263P:

        case AV_CODEC_ID_FLV1:

            if (CONFIG_H263_ENCODER)

                ff_clean_h263_qscales(s);

            break;

        default:

            ff_init_qscale_tab(s);

        }



        s->lambda= s->lambda_table[0];

        //FIXME broken

    }else

        s->lambda = s->current_picture.f.quality;

    update_qscale(s);

    return 0;

}
