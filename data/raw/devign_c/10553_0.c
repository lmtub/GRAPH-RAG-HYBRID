static int mov_read_mvhd(MOVContext *c, AVIOContext *pb, MOVAtom atom)

{

    int i;

    int64_t creation_time;

    int version = avio_r8(pb); /* version */

    avio_rb24(pb); /* flags */



    if (version == 1) {

        creation_time = avio_rb64(pb);

        avio_rb64(pb);

    } else {

        creation_time = avio_rb32(pb);

        avio_rb32(pb); /* modification time */

    }

    mov_metadata_creation_time(&c->fc->metadata, creation_time);

    c->time_scale = avio_rb32(pb); /* time scale */

    if (c->time_scale <= 0) {

        av_log(c->fc, AV_LOG_ERROR, "Invalid mvhd time scale %d\n", c->time_scale);

        return AVERROR_INVALIDDATA;

    }

    av_log(c->fc, AV_LOG_TRACE, "time scale = %i\n", c->time_scale);



    c->duration = (version == 1) ? avio_rb64(pb) : avio_rb32(pb); /* duration */

    // set the AVCodecContext duration because the duration of individual tracks

    // may be inaccurate

    if (c->time_scale > 0 && !c->trex_data)

        c->fc->duration = av_rescale(c->duration, AV_TIME_BASE, c->time_scale);

    avio_rb32(pb); /* preferred scale */



    avio_rb16(pb); /* preferred volume */



    avio_skip(pb, 10); /* reserved */



    /* movie display matrix, store it in main context and use it later on */

    for (i = 0; i < 3; i++) {

        c->movie_display_matrix[i][0] = avio_rb32(pb); // 16.16 fixed point

        c->movie_display_matrix[i][1] = avio_rb32(pb); // 16.16 fixed point

        c->movie_display_matrix[i][2] = avio_rb32(pb); //  2.30 fixed point

    }



    avio_rb32(pb); /* preview time */

    avio_rb32(pb); /* preview duration */

    avio_rb32(pb); /* poster time */

    avio_rb32(pb); /* selection time */

    avio_rb32(pb); /* selection duration */

    avio_rb32(pb); /* current time */

    avio_rb32(pb); /* next track ID */



    return 0;

}
