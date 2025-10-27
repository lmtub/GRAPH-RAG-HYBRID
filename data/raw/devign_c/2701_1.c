void ff_h264_direct_dist_scale_factor(H264Context * const h){

    const int poc = h->cur_pic_ptr->field_poc[h->picture_structure == PICT_BOTTOM_FIELD];

    const int poc1 = h->ref_list[1][0].poc;

    int i, field;



    if (FRAME_MBAFF(h))

        for (field = 0; field < 2; field++){

            const int poc  = h->cur_pic_ptr->field_poc[field];

            const int poc1 = h->ref_list[1][0].field_poc[field];

            for (i = 0; i < 2 * h->ref_count[0]; i++)

                h->dist_scale_factor_field[field][i^field] =

                    get_scale_factor(h, poc, poc1, i+16);

        }



    for (i = 0; i < h->ref_count[0]; i++){

        h->dist_scale_factor[i] = get_scale_factor(h, poc, poc1, i);

    }

}
