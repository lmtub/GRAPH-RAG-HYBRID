static int frame_start(MpegEncContext *s)

{

    int ret;



    /* mark & release old frames */

    if (s->pict_type != AV_PICTURE_TYPE_B && s->last_picture_ptr &&

        s->last_picture_ptr != s->next_picture_ptr &&

        s->last_picture_ptr->f.buf[0]) {

        ff_mpeg_unref_picture(s, s->last_picture_ptr);

    }



    s->current_picture_ptr->f.pict_type = s->pict_type;

    s->current_picture_ptr->f.key_frame = s->pict_type == AV_PICTURE_TYPE_I;



    ff_mpeg_unref_picture(s, &s->current_picture);

    if ((ret = ff_mpeg_ref_picture(s, &s->current_picture,

                                   s->current_picture_ptr)) < 0)

        return ret;



    if (s->pict_type != AV_PICTURE_TYPE_B) {

        s->last_picture_ptr = s->next_picture_ptr;

        if (!s->droppable)

            s->next_picture_ptr = s->current_picture_ptr;

    }



    if (s->last_picture_ptr) {

        ff_mpeg_unref_picture(s, &s->last_picture);

        if (s->last_picture_ptr->f.buf[0] &&

            (ret = ff_mpeg_ref_picture(s, &s->last_picture,

                                       s->last_picture_ptr)) < 0)

            return ret;

    }

    if (s->next_picture_ptr) {

        ff_mpeg_unref_picture(s, &s->next_picture);

        if (s->next_picture_ptr->f.buf[0] &&

            (ret = ff_mpeg_ref_picture(s, &s->next_picture,

                                       s->next_picture_ptr)) < 0)

            return ret;

    }



    if (s->picture_structure!= PICT_FRAME) {

        int i;

        for (i = 0; i < 4; i++) {

            if (s->picture_structure == PICT_BOTTOM_FIELD) {

                s->current_picture.f.data[i] +=

                    s->current_picture.f.linesize[i];

            }

            s->current_picture.f.linesize[i] *= 2;

            s->last_picture.f.linesize[i]    *= 2;

            s->next_picture.f.linesize[i]    *= 2;

        }

    }



    if (s->mpeg_quant || s->codec_id == AV_CODEC_ID_MPEG2VIDEO) {

        s->dct_unquantize_intra = s->dct_unquantize_mpeg2_intra;

        s->dct_unquantize_inter = s->dct_unquantize_mpeg2_inter;

    } else if (s->out_format == FMT_H263 || s->out_format == FMT_H261) {

        s->dct_unquantize_intra = s->dct_unquantize_h263_intra;

        s->dct_unquantize_inter = s->dct_unquantize_h263_inter;

    } else {

        s->dct_unquantize_intra = s->dct_unquantize_mpeg1_intra;

        s->dct_unquantize_inter = s->dct_unquantize_mpeg1_inter;

    }



    if (s->dct_error_sum) {

        assert(s->avctx->noise_reduction && s->encoding);

        update_noise_reduction(s);

    }



    return 0;

}
