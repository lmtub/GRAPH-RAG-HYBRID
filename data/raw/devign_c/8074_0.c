static int get_cluster_duration(MOVTrack *track, int cluster_idx)

{

    int64_t next_dts;



    if (cluster_idx >= track->entry)

        return 0;



    if (cluster_idx + 1 == track->entry)

        next_dts = track->track_duration + track->start_dts;

    else

        next_dts = track->cluster[cluster_idx + 1].dts;



    return next_dts - track->cluster[cluster_idx].dts;

}
