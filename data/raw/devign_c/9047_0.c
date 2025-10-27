static int write_fragments(struct Tracks *tracks, int start_index,

                           AVIOContext *in)

{

    char dirname[100], filename[500];

    int i, j;



    for (i = start_index; i < tracks->nb_tracks; i++) {

        struct Track *track = tracks->tracks[i];

        const char *type    = track->is_video ? "video" : "audio";

        snprintf(dirname, sizeof(dirname), "QualityLevels(%d)", track->bitrate);

        mkdir(dirname, 0777);

        for (j = 0; j < track->chunks; j++) {

            snprintf(filename, sizeof(filename), "%s/Fragments(%s=%"PRId64")",

                     dirname, type, track->offsets[j].time);

            avio_seek(in, track->offsets[j].offset, SEEK_SET);

            write_fragment(filename, in);

        }

    }

    return 0;

}
