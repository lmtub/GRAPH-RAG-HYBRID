static void pmt_cb(MpegTSFilter *filter, const uint8_t *section, int section_len)

{

    MpegTSContext *ts = filter->u.section_filter.opaque;

    MpegTSSectionFilter *tssf = &filter->u.section_filter;

    SectionHeader h1, *h = &h1;

    PESContext *pes;

    AVStream *st;

    const uint8_t *p, *p_end, *desc_list_end;

    int program_info_length, pcr_pid, pid, stream_type;

    int desc_list_len;

    uint32_t prog_reg_desc = 0; /* registration descriptor */



    int mp4_descr_count = 0;

    Mp4Descr mp4_descr[MAX_MP4_DESCR_COUNT] = { { 0 } };

    int i;



    av_log(ts->stream, AV_LOG_TRACE, "PMT: len %i\n", section_len);

    hex_dump_debug(ts->stream, section, section_len);



    p_end = section + section_len - 4;

    p = section;

    if (parse_section_header(h, &p, p_end) < 0)

        return;

    if (h->version == tssf->last_ver)

        return;

    tssf->last_ver = h->version;



    av_log(ts->stream, AV_LOG_TRACE, "sid=0x%x sec_num=%d/%d version=%d\n",

            h->id, h->sec_num, h->last_sec_num, h->version);



    if (h->tid != PMT_TID)

        return;

    if (!ts->scan_all_pmts && ts->skip_changes)

        return;



    if (!ts->skip_clear)

        clear_program(ts, h->id);



    pcr_pid = get16(&p, p_end);

    if (pcr_pid < 0)

        return;

    pcr_pid &= 0x1fff;

    add_pid_to_pmt(ts, h->id, pcr_pid);

    set_pcr_pid(ts->stream, h->id, pcr_pid);



    av_log(ts->stream, AV_LOG_TRACE, "pcr_pid=0x%x\n", pcr_pid);



    program_info_length = get16(&p, p_end);

    if (program_info_length < 0)

        return;

    program_info_length &= 0xfff;

    while (program_info_length >= 2) {

        uint8_t tag, len;

        tag = get8(&p, p_end);

        len = get8(&p, p_end);



        av_log(ts->stream, AV_LOG_TRACE, "program tag: 0x%02x len=%d\n", tag, len);



        if (len > program_info_length - 2)

            // something else is broken, exit the program_descriptors_loop

            break;

        program_info_length -= len + 2;

        if (tag == 0x1d) { // IOD descriptor

            get8(&p, p_end); // scope

            get8(&p, p_end); // label

            len -= 2;

            mp4_read_iods(ts->stream, p, len, mp4_descr + mp4_descr_count,

                          &mp4_descr_count, MAX_MP4_DESCR_COUNT);

        } else if (tag == 0x05 && len >= 4) { // registration descriptor

            prog_reg_desc = bytestream_get_le32(&p);

            len -= 4;

        }

        p += len;

    }

    p += program_info_length;

    if (p >= p_end)

        goto out;



    // stop parsing after pmt, we found header

    if (!ts->stream->nb_streams)

        ts->stop_parse = 2;



    set_pmt_found(ts, h->id);





    for (;;) {

        st = 0;

        pes = NULL;

        stream_type = get8(&p, p_end);

        if (stream_type < 0)

            break;

        pid = get16(&p, p_end);

        if (pid < 0)

            goto out;

        pid &= 0x1fff;

        if (pid == ts->current_pid)

            goto out;



        /* now create stream */

        if (ts->pids[pid] && ts->pids[pid]->type == MPEGTS_PES) {

            pes = ts->pids[pid]->u.pes_filter.opaque;

            if (!pes->st) {

                pes->st     = avformat_new_stream(pes->stream, NULL);

                if (!pes->st)

                    goto out;

                pes->st->id = pes->pid;

            }

            st = pes->st;

        } else if (stream_type != 0x13) {

            if (ts->pids[pid])

                mpegts_close_filter(ts, ts->pids[pid]); // wrongly added sdt filter probably

            pes = add_pes_stream(ts, pid, pcr_pid);

            if (pes) {

                st = avformat_new_stream(pes->stream, NULL);

                if (!st)

                    goto out;

                st->id = pes->pid;

            }

        } else {

            int idx = ff_find_stream_index(ts->stream, pid);

            if (idx >= 0) {

                st = ts->stream->streams[idx];

            } else {

                st = avformat_new_stream(ts->stream, NULL);

                if (!st)

                    goto out;

                st->id = pid;

                st->codec->codec_type = AVMEDIA_TYPE_DATA;

            }

        }



        if (!st)

            goto out;



        if (pes && !pes->stream_type)

            mpegts_set_stream_info(st, pes, stream_type, prog_reg_desc);



        add_pid_to_pmt(ts, h->id, pid);



        ff_program_add_stream_index(ts->stream, h->id, st->index);



        desc_list_len = get16(&p, p_end);

        if (desc_list_len < 0)

            goto out;

        desc_list_len &= 0xfff;

        desc_list_end  = p + desc_list_len;

        if (desc_list_end > p_end)

            goto out;

        for (;;) {

            if (ff_parse_mpeg2_descriptor(ts->stream, st, stream_type, &p,

                                          desc_list_end, mp4_descr,

                                          mp4_descr_count, pid, ts) < 0)

                break;



            if (pes && prog_reg_desc == AV_RL32("HDMV") &&

                stream_type == 0x83 && pes->sub_st) {

                ff_program_add_stream_index(ts->stream, h->id,

                                            pes->sub_st->index);

                pes->sub_st->codec->codec_tag = st->codec->codec_tag;

            }

        }

        p = desc_list_end;

    }



    if (!ts->pids[pcr_pid])

        mpegts_open_pcr_filter(ts, pcr_pid);



out:

    for (i = 0; i < mp4_descr_count; i++)

        av_free(mp4_descr[i].dec_config_descr);

}
