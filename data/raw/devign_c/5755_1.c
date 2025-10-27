void ff_h264_flush_change(H264Context *h)

{

    int i, j;



    h->next_outputed_poc = INT_MIN;

    h->prev_interlaced_frame = 1;

    idr(h);



    h->poc.prev_frame_num = -1;

    if (h->cur_pic_ptr) {

        h->cur_pic_ptr->reference = 0;

        for (j=i=0; h->delayed_pic[i]; i++)

            if (h->delayed_pic[i] != h->cur_pic_ptr)

                h->delayed_pic[j++] = h->delayed_pic[i];

        h->delayed_pic[j] = NULL;

    }

    ff_h264_unref_picture(h, &h->last_pic_for_ec);



    h->first_field = 0;

    ff_h264_sei_uninit(&h->sei);

    h->recovery_frame = -1;

    h->frame_recovered = 0;

    h->current_slice = 0;

    h->mmco_reset = 1;

    for (i = 0; i < h->nb_slice_ctx; i++)

        h->slice_ctx[i].list_count = 0;

}
