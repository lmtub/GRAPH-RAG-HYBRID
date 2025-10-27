static void write_section_data(MpegTSContext *ts, MpegTSFilter *tss1,

                               const uint8_t *buf, int buf_size, int is_start)

{

    MpegTSSectionFilter *tss = &tss1->u.section_filter;

    int len;



    if (is_start) {

        memcpy(tss->section_buf, buf, buf_size);

        tss->section_index = buf_size;

        tss->section_h_size = -1;

        tss->end_of_section_reached = 0;

    } else {

        if (tss->end_of_section_reached)

            return;

        len = 4096 - tss->section_index;

        if (buf_size < len)

            len = buf_size;

        memcpy(tss->section_buf + tss->section_index, buf, len);

        tss->section_index += len;

    }



    /* compute section length if possible */

    if (tss->section_h_size == -1 && tss->section_index >= 3) {

        len = (AV_RB16(tss->section_buf + 1) & 0xfff) + 3;

        if (len > 4096)

            return;

        tss->section_h_size = len;

    }



    if (tss->section_h_size != -1 &&

        tss->section_index >= tss->section_h_size) {

        int crc_valid = 1;

        tss->end_of_section_reached = 1;



        if (tss->check_crc) {

            crc_valid = !av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, tss->section_buf, tss->section_h_size);

            if (crc_valid) {

                ts->crc_validity[ tss1->pid ] = 100;

            }else if (ts->crc_validity[ tss1->pid ] > -10) {

                ts->crc_validity[ tss1->pid ]--;

            }else

                crc_valid = 2;

        }

        if (crc_valid)

            tss->section_cb(tss1, tss->section_buf, tss->section_h_size);

    }

}
