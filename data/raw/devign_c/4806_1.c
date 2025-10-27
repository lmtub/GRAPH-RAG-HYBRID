static int h264_slice_init(H264Context *h, H264SliceContext *sl,

                           const H2645NAL *nal)

{

    int i, j, ret = 0;



    if (h->picture_idr && nal->type != H264_NAL_IDR_SLICE) {

        av_log(h->avctx, AV_LOG_ERROR, "Invalid mix of IDR and non-IDR slices\n");

        return AVERROR_INVALIDDATA;

    }



    av_assert1(h->mb_num == h->mb_width * h->mb_height);

    if (sl->first_mb_addr << FIELD_OR_MBAFF_PICTURE(h) >= h->mb_num ||

        sl->first_mb_addr >= h->mb_num) {

        av_log(h->avctx, AV_LOG_ERROR, "first_mb_in_slice overflow\n");

        return AVERROR_INVALIDDATA;

    }

    sl->resync_mb_x = sl->mb_x =  sl->first_mb_addr % h->mb_width;

    sl->resync_mb_y = sl->mb_y = (sl->first_mb_addr / h->mb_width) <<

                                 FIELD_OR_MBAFF_PICTURE(h);

    if (h->picture_structure == PICT_BOTTOM_FIELD)

        sl->resync_mb_y = sl->mb_y = sl->mb_y + 1;

    av_assert1(sl->mb_y < h->mb_height);



    ret = ff_h264_build_ref_list(h, sl);

    if (ret < 0)

        return ret;



    if (h->ps.pps->weighted_bipred_idc == 2 &&

        sl->slice_type_nos == AV_PICTURE_TYPE_B) {

        implicit_weight_table(h, sl, -1);

        if (FRAME_MBAFF(h)) {

            implicit_weight_table(h, sl, 0);

            implicit_weight_table(h, sl, 1);

        }

    }



    if (sl->slice_type_nos == AV_PICTURE_TYPE_B && !sl->direct_spatial_mv_pred)

        ff_h264_direct_dist_scale_factor(h, sl);

    ff_h264_direct_ref_list_init(h, sl);



    if (h->avctx->skip_loop_filter >= AVDISCARD_ALL ||

        (h->avctx->skip_loop_filter >= AVDISCARD_NONKEY &&

         h->nal_unit_type != H264_NAL_IDR_SLICE) ||

        (h->avctx->skip_loop_filter >= AVDISCARD_NONINTRA &&

         sl->slice_type_nos != AV_PICTURE_TYPE_I) ||

        (h->avctx->skip_loop_filter >= AVDISCARD_BIDIR  &&

         sl->slice_type_nos == AV_PICTURE_TYPE_B) ||

        (h->avctx->skip_loop_filter >= AVDISCARD_NONREF &&

         nal->ref_idc == 0))

        sl->deblocking_filter = 0;



    if (sl->deblocking_filter == 1 && h->nb_slice_ctx > 1) {

        if (h->avctx->flags2 & AV_CODEC_FLAG2_FAST) {

            /* Cheat slightly for speed:

             * Do not bother to deblock across slices. */

            sl->deblocking_filter = 2;

        } else {

            h->postpone_filter = 1;

        }

    }

    sl->qp_thresh = 15 -

                   FFMIN(sl->slice_alpha_c0_offset, sl->slice_beta_offset) -

                   FFMAX3(0,

                          h->ps.pps->chroma_qp_index_offset[0],

                          h->ps.pps->chroma_qp_index_offset[1]) +

                   6 * (h->ps.sps->bit_depth_luma - 8);



    sl->slice_num       = ++h->current_slice;



    if (sl->slice_num)

        h->slice_row[(sl->slice_num-1)&(MAX_SLICES-1)]= sl->resync_mb_y;

    if (   h->slice_row[sl->slice_num&(MAX_SLICES-1)] + 3 >= sl->resync_mb_y

        && h->slice_row[sl->slice_num&(MAX_SLICES-1)] <= sl->resync_mb_y

        && sl->slice_num >= MAX_SLICES) {

        //in case of ASO this check needs to be updated depending on how we decide to assign slice numbers in this case

        av_log(h->avctx, AV_LOG_WARNING, "Possibly too many slices (%d >= %d), increase MAX_SLICES and recompile if there are artifacts\n", sl->slice_num, MAX_SLICES);

    }



    for (j = 0; j < 2; j++) {

        int id_list[16];

        int *ref2frm = h->ref2frm[sl->slice_num & (MAX_SLICES - 1)][j];

        for (i = 0; i < 16; i++) {

            id_list[i] = 60;

            if (j < sl->list_count && i < sl->ref_count[j] &&

                sl->ref_list[j][i].parent->f->buf[0]) {

                int k;

                AVBuffer *buf = sl->ref_list[j][i].parent->f->buf[0]->buffer;

                for (k = 0; k < h->short_ref_count; k++)

                    if (h->short_ref[k]->f->buf[0]->buffer == buf) {

                        id_list[i] = k;

                        break;

                    }

                for (k = 0; k < h->long_ref_count; k++)

                    if (h->long_ref[k] && h->long_ref[k]->f->buf[0]->buffer == buf) {

                        id_list[i] = h->short_ref_count + k;

                        break;

                    }

            }

        }



        ref2frm[0] =

        ref2frm[1] = -1;

        for (i = 0; i < 16; i++)

            ref2frm[i + 2] = 4 * id_list[i] + (sl->ref_list[j][i].reference & 3);

        ref2frm[18 + 0] =

        ref2frm[18 + 1] = -1;

        for (i = 16; i < 48; i++)

            ref2frm[i + 4] = 4 * id_list[(i - 16) >> 1] +

                             (sl->ref_list[j][i].reference & 3);

    }



    if (h->avctx->debug & FF_DEBUG_PICT_INFO) {

        av_log(h->avctx, AV_LOG_DEBUG,

               "slice:%d %s mb:%d %c%s%s frame:%d poc:%d/%d ref:%d/%d qp:%d loop:%d:%d:%d weight:%d%s %s\n",

               sl->slice_num,

               (h->picture_structure == PICT_FRAME ? "F" : h->picture_structure == PICT_TOP_FIELD ? "T" : "B"),

               sl->mb_y * h->mb_width + sl->mb_x,

               av_get_picture_type_char(sl->slice_type),

               sl->slice_type_fixed ? " fix" : "",

               nal->type == H264_NAL_IDR_SLICE ? " IDR" : "",

               h->poc.frame_num,

               h->cur_pic_ptr->field_poc[0],

               h->cur_pic_ptr->field_poc[1],

               sl->ref_count[0], sl->ref_count[1],

               sl->qscale,

               sl->deblocking_filter,

               sl->slice_alpha_c0_offset, sl->slice_beta_offset,

               sl->pwt.use_weight,

               sl->pwt.use_weight == 1 && sl->pwt.use_weight_chroma ? "c" : "",

               sl->slice_type == AV_PICTURE_TYPE_B ? (sl->direct_spatial_mv_pred ? "SPAT" : "TEMP") : "");

    }



    return 0;

}
