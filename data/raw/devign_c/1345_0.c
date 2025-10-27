static int64_t mkv_write_seekhead(AVIOContext *pb, MatroskaMuxContext *mkv)

{

    AVIOContext *dyn_cp;

    mkv_seekhead *seekhead = mkv->main_seekhead;

    ebml_master metaseek, seekentry;

    int64_t currentpos;

    int i;



    currentpos = avio_tell(pb);



    if (seekhead->reserved_size > 0) {

        if (avio_seek(pb, seekhead->filepos, SEEK_SET) < 0) {

            currentpos = -1;

            goto fail;

        }

    }



    if (start_ebml_master_crc32(pb, &dyn_cp, &metaseek, MATROSKA_ID_SEEKHEAD,

                                seekhead->reserved_size) < 0) {

        currentpos = -1;

        goto fail;

    }



    for (i = 0; i < seekhead->num_entries; i++) {

        mkv_seekhead_entry *entry = &seekhead->entries[i];



        seekentry = start_ebml_master(dyn_cp, MATROSKA_ID_SEEKENTRY, MAX_SEEKENTRY_SIZE);



        put_ebml_id(dyn_cp, MATROSKA_ID_SEEKID);

        put_ebml_num(dyn_cp, ebml_id_size(entry->elementid), 0);

        put_ebml_id(dyn_cp, entry->elementid);



        put_ebml_uint(dyn_cp, MATROSKA_ID_SEEKPOSITION, entry->segmentpos);

        end_ebml_master(dyn_cp, seekentry);

    }

    end_ebml_master_crc32(pb, &dyn_cp, mkv, metaseek);



    if (seekhead->reserved_size > 0) {

        uint64_t remaining = seekhead->filepos + seekhead->reserved_size - avio_tell(pb);

        put_ebml_void(pb, remaining);

        avio_seek(pb, currentpos, SEEK_SET);



        currentpos = seekhead->filepos;

    }

fail:

    av_freep(&mkv->main_seekhead->entries);

    av_freep(&mkv->main_seekhead);



    return currentpos;

}
