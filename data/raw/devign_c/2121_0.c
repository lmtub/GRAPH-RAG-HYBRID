void ff_generate_sliding_window_mmcos(H264Context *h, int first_slice)

{

    MpegEncContext * const s = &h->s;

    MMCO mmco_temp[MAX_MMCO_COUNT], *mmco = first_slice ? h->mmco : mmco_temp;

    int mmco_index = 0, i;



    assert(h->long_ref_count + h->short_ref_count <= h->sps.ref_frame_count);



    if (h->short_ref_count &&

        h->long_ref_count + h->short_ref_count == h->sps.ref_frame_count &&

        !(FIELD_PICTURE && !s->first_field &&

          s->current_picture_ptr->f.reference)) {

        mmco[0].opcode = MMCO_SHORT2UNUSED;

        mmco[0].short_pic_num = h->short_ref[h->short_ref_count - 1]->frame_num;

        mmco_index = 1;

        if (FIELD_PICTURE) {

            mmco[0].short_pic_num *= 2;

            mmco[1].opcode = MMCO_SHORT2UNUSED;

            mmco[1].short_pic_num = mmco[0].short_pic_num + 1;

            mmco_index = 2;

        }

    }



    if (first_slice) {

        h->mmco_index = mmco_index;

    } else if (!first_slice && mmco_index >= 0 &&

               (mmco_index != h->mmco_index ||

                (i = check_opcodes(h->mmco, mmco_temp, mmco_index)))) {

        av_log(h->s.avctx, AV_LOG_ERROR,

               "Inconsistent MMCO state between slices [%d, %d, %d]\n",

               mmco_index, h->mmco_index, i);

        return AVERROR_INVALIDDATA;

    }

}
