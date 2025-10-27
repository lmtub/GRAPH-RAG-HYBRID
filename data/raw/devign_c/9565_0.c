static int mov_write_trailer(AVFormatContext *s)

{

    MOVMuxContext *mov = s->priv_data;

    AVIOContext *pb = s->pb;

    int res = 0;

    int i;

    int64_t moov_pos;



    /*

     * Before actually writing the trailer, make sure that there are no

     * dangling subtitles, that need a terminating sample.

     */

    for (i = 0; i < mov->nb_streams; i++) {

        MOVTrack *trk = &mov->tracks[i];

        if (trk->enc->codec_id == AV_CODEC_ID_MOV_TEXT &&

            !trk->last_sample_is_subtitle_end) {

            mov_write_subtitle_end_packet(s, i, trk->track_duration);

            trk->last_sample_is_subtitle_end = 1;

        }

    }



    // If there were no chapters when the header was written, but there

    // are chapters now, write them in the trailer.  This only works

    // when we are not doing fragments.

    if (!mov->chapter_track && !(mov->flags & FF_MOV_FLAG_FRAGMENT)) {

        if (mov->mode & (MODE_MP4|MODE_MOV|MODE_IPOD) && s->nb_chapters) {

            mov->chapter_track = mov->nb_streams++;

            if ((res = mov_create_chapter_track(s, mov->chapter_track)) < 0)

                goto error;

        }

    }



    if (!(mov->flags & FF_MOV_FLAG_FRAGMENT)) {

        moov_pos = avio_tell(pb);



        /* Write size of mdat tag */

        if (mov->mdat_size + 8 <= UINT32_MAX) {

            avio_seek(pb, mov->mdat_pos, SEEK_SET);

            avio_wb32(pb, mov->mdat_size + 8);

        } else {

            /* overwrite 'wide' placeholder atom */

            avio_seek(pb, mov->mdat_pos - 8, SEEK_SET);

            /* special value: real atom size will be 64 bit value after

             * tag field */

            avio_wb32(pb, 1);

            ffio_wfourcc(pb, "mdat");

            avio_wb64(pb, mov->mdat_size + 16);

        }

        avio_seek(pb, mov->reserved_moov_size > 0 ? mov->reserved_moov_pos : moov_pos, SEEK_SET);



        if (mov->flags & FF_MOV_FLAG_FASTSTART) {

            av_log(s, AV_LOG_INFO, "Starting second pass: moving the moov atom to the beginning of the file\n");

            res = shift_data(s);

            if (res == 0) {

                avio_seek(s->pb, mov->reserved_moov_pos, SEEK_SET);

                mov_write_moov_tag(pb, mov, s);

            }

        } else if (mov->reserved_moov_size > 0) {

            int64_t size;

            mov_write_moov_tag(pb, mov, s);

            size = mov->reserved_moov_size - (avio_tell(pb) - mov->reserved_moov_pos);

            if (size < 8){

                av_log(s, AV_LOG_ERROR, "reserved_moov_size is too small, needed %"PRId64" additional\n", 8-size);

                return -1;

            }

            avio_wb32(pb, size);

            ffio_wfourcc(pb, "free");

            for (i = 0; i < size; i++)

                avio_w8(pb, 0);

            avio_seek(pb, moov_pos, SEEK_SET);

        } else {

            mov_write_moov_tag(pb, mov, s);

        }

    } else {

        mov_flush_fragment(s);

        mov_write_mfra_tag(pb, mov);

    }



    for (i = 0; i < mov->nb_streams; i++) {

        if (mov->flags & FF_MOV_FLAG_FRAGMENT &&

            mov->tracks[i].vc1_info.struct_offset && s->pb->seekable) {

            int64_t off = avio_tell(pb);

            uint8_t buf[7];

            if (mov_write_dvc1_structs(&mov->tracks[i], buf) >= 0) {

                avio_seek(pb, mov->tracks[i].vc1_info.struct_offset, SEEK_SET);

                avio_write(pb, buf, 7);

                avio_seek(pb, off, SEEK_SET);

            }

        }

    }



error:

    mov_free(s);



    return res;

}
