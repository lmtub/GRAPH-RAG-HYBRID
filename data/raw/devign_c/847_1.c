static int cin_read_frame_header(CinDemuxContext *cin, AVIOContext *pb) {

    CinFrameHeader *hdr = &cin->frame_header;



    hdr->video_frame_type = avio_r8(pb);

    hdr->audio_frame_type = avio_r8(pb);

    hdr->pal_colors_count = avio_rl16(pb);

    hdr->video_frame_size = avio_rl32(pb);

    hdr->audio_frame_size = avio_rl32(pb);



    if (pb->eof_reached || pb->error)

        return AVERROR(EIO);



    if (avio_rl32(pb) != 0xAA55AA55)






    return 0;

}