static int mp3_write_trailer(struct AVFormatContext *s)

{

    uint8_t buf[ID3v1_TAG_SIZE];

    MP3Context *mp3 = s->priv_data;



    /* write the id3v1 tag */

    if (id3v1_create_tag(s, buf) > 0) {

        avio_write(s->pb, buf, ID3v1_TAG_SIZE);

    }



    /* write number of frames */

    if (mp3 && mp3->nb_frames_offset) {

        avio_seek(s->pb, mp3->nb_frames_offset, SEEK_SET);

        avio_wb32(s->pb, s->streams[0]->nb_frames);

        avio_seek(s->pb, 0, SEEK_END);

    }



    avio_flush(s->pb);



    return 0;

}
