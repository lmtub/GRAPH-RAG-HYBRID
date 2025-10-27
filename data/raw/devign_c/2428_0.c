void ff_generate_sliding_window_mmcos(H264Context *h) {

    MpegEncContext * const s = &h->s;

    assert(h->long_ref_count + h->short_ref_count <= h->sps.ref_frame_count);



    h->mmco_index= 0;

    if(h->short_ref_count && h->long_ref_count + h->short_ref_count == h->sps.ref_frame_count &&

            !(FIELD_PICTURE && !s->first_field && s->current_picture_ptr->reference)) {

        h->mmco[0].opcode= MMCO_SHORT2UNUSED;

        h->mmco[0].short_pic_num= h->short_ref[ h->short_ref_count - 1 ]->frame_num;

        h->mmco_index= 1;

        if (FIELD_PICTURE) {

            h->mmco[0].short_pic_num *= 2;

            h->mmco[1].opcode= MMCO_SHORT2UNUSED;

            h->mmco[1].short_pic_num= h->mmco[0].short_pic_num + 1;

            h->mmco_index= 2;

        }

    }

}
