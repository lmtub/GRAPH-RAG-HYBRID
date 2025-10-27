static int mxf_write_footer(AVFormatContext *s)
{
    MXFContext *mxf = s->priv_data;
    AVIOContext *pb = s->pb;
    int err = 0;
    mxf->duration = mxf->last_indexed_edit_unit + mxf->edit_units_count;
    mxf_write_klv_fill(s);
    mxf->footer_partition_offset = avio_tell(pb);
    if (mxf->edit_unit_byte_count && s->oformat != &ff_mxf_opatom_muxer) { // no need to repeat index
        if ((err = mxf_write_partition(s, 0, 0, footer_partition_key, 0)) < 0)
    } else {
        if ((err = mxf_write_partition(s, 0, 2, footer_partition_key, 0)) < 0)
        mxf_write_klv_fill(s);
        mxf_write_index_table_segment(s);
    mxf_write_klv_fill(s);
    mxf_write_random_index_pack(s);
    if (s->pb->seekable) {
        if (s->oformat == &ff_mxf_opatom_muxer){
            /* rewrite body partition to update lengths */
            avio_seek(pb, mxf->body_partition_offset[0], SEEK_SET);
            if ((err = mxf_write_opatom_body_partition(s)) < 0)
        avio_seek(pb, 0, SEEK_SET);
        if (mxf->edit_unit_byte_count && s->oformat != &ff_mxf_opatom_muxer) {
            if ((err = mxf_write_partition(s, 1, 2, header_closed_partition_key, 1)) < 0)
            mxf_write_klv_fill(s);
            mxf_write_index_table_segment(s);
        } else {
            if ((err = mxf_write_partition(s, 0, 0, header_closed_partition_key, 1)) < 0)
end:
    ff_audio_interleave_close(s);
    av_freep(&mxf->index_entries);
    av_freep(&mxf->body_partition_offset);
    av_freep(&mxf->timecode_track->priv_data);
    av_freep(&mxf->timecode_track);
    mxf_free(s);
    return err < 0 ? err : 0;