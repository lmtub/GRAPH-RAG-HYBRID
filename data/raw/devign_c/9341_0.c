static int mov_write_source_reference_tag(AVIOContext *pb, MOVTrack *track, const char *reel_name){

    int64_t pos = avio_tell(pb);

    avio_wb32(pb, 0);                              /* size */

    ffio_wfourcc(pb, "name");                      /* Data format */

    avio_wb16(pb, strlen(reel_name));              /* string size */

    avio_wb16(pb, track->language);                /* langcode */

    avio_write(pb, reel_name, strlen(reel_name));  /* reel name */

    return update_size(pb,pos);

}
