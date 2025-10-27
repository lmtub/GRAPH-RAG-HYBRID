static void mp3_update_xing(AVFormatContext *s)

{

    MP3Context  *mp3 = s->priv_data;

    int i;



    /* replace "Xing" identification string with "Info" for CBR files. */

    if (!mp3->has_variable_bitrate) {

        avio_seek(s->pb, mp3->xing_offset, SEEK_SET);

        ffio_wfourcc(s->pb, "Info");

    }



    avio_seek(s->pb, mp3->xing_offset + 8, SEEK_SET);

    avio_wb32(s->pb, mp3->frames);

    avio_wb32(s->pb, mp3->size);



    avio_w8(s->pb, 0);  // first toc entry has to be zero.



    for (i = 1; i < XING_TOC_SIZE; ++i) {

        int j = i * mp3->pos / XING_TOC_SIZE;

        int seek_point = 256LL * mp3->bag[j] / mp3->size;

        avio_w8(s->pb, FFMIN(seek_point, 255));

    }



    avio_seek(s->pb, 0, SEEK_END);

}
