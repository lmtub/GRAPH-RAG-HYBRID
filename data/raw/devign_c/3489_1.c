void ff_h264_direct_ref_list_init(const H264Context *const h, H264SliceContext *sl)

{

    H264Ref *const ref1 = &sl->ref_list[1][0];

    H264Picture *const cur = h->cur_pic_ptr;

    int list, j, field;

    int sidx     = (h->picture_structure & 1) ^ 1;

    int ref1sidx = (ref1->reference      & 1) ^ 1;



    for (list = 0; list < sl->list_count; list++) {

        cur->ref_count[sidx][list] = sl->ref_count[list];

        for (j = 0; j < sl->ref_count[list]; j++)

            cur->ref_poc[sidx][list][j] = 4 * sl->ref_list[list][j].parent->frame_num +

                                          (sl->ref_list[list][j].reference & 3);

    }



    if (h->picture_structure == PICT_FRAME) {

        memcpy(cur->ref_count[1], cur->ref_count[0], sizeof(cur->ref_count[0]));

        memcpy(cur->ref_poc[1],   cur->ref_poc[0],   sizeof(cur->ref_poc[0]));

    }



    cur->mbaff = FRAME_MBAFF(h);



    sl->col_fieldoff = 0;



    if (sl->list_count != 2 || !sl->ref_count[1])

        return;



    if (h->picture_structure == PICT_FRAME) {

        int cur_poc  = h->cur_pic_ptr->poc;

        int *col_poc = sl->ref_list[1][0].parent->field_poc;





        sl->col_parity = (FFABS(col_poc[0] - cur_poc) >=

                          FFABS(col_poc[1] - cur_poc));

        ref1sidx =

        sidx     = sl->col_parity;

    // FL -> FL & differ parity

    } else if (!(h->picture_structure & sl->ref_list[1][0].reference) &&

               !sl->ref_list[1][0].parent->mbaff) {

        sl->col_fieldoff = 2 * sl->ref_list[1][0].reference - 3;

    }



    if (sl->slice_type_nos != AV_PICTURE_TYPE_B || sl->direct_spatial_mv_pred)

        return;



    for (list = 0; list < 2; list++) {

        fill_colmap(h, sl, sl->map_col_to_list0, list, sidx, ref1sidx, 0);

        if (FRAME_MBAFF(h))

            for (field = 0; field < 2; field++)

                fill_colmap(h, sl, sl->map_col_to_list0_field[field], list, field,

                            field, 1);

    }

}