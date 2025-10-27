static int64_t mkv_write_cues(AVIOContext *pb, mkv_cues *cues, int num_tracks)

{

    ebml_master cues_element;

    int64_t currentpos;

    int i, j;



    currentpos = avio_tell(pb);

    cues_element = start_ebml_master(pb, MATROSKA_ID_CUES, 0);



    for (i = 0; i < cues->num_entries; i++) {

        ebml_master cuepoint, track_positions;

        mkv_cuepoint *entry = &cues->entries[i];

        uint64_t pts = entry->pts;



        cuepoint = start_ebml_master(pb, MATROSKA_ID_POINTENTRY, MAX_CUEPOINT_SIZE(num_tracks));

        put_ebml_uint(pb, MATROSKA_ID_CUETIME, pts);



        // put all the entries from different tracks that have the exact same

        // timestamp into the same CuePoint

        for (j = 0; j < cues->num_entries - i && entry[j].pts == pts; j++) {

            track_positions = start_ebml_master(pb, MATROSKA_ID_CUETRACKPOSITION, MAX_CUETRACKPOS_SIZE);

            put_ebml_uint(pb, MATROSKA_ID_CUETRACK          , entry[j].tracknum   );

            put_ebml_uint(pb, MATROSKA_ID_CUECLUSTERPOSITION, entry[j].cluster_pos);

            end_ebml_master(pb, track_positions);

        }

        i += j - 1;

        end_ebml_master(pb, cuepoint);

    }

    end_ebml_master(pb, cues_element);



    av_free(cues->entries);

    av_free(cues);

    return currentpos;

}
