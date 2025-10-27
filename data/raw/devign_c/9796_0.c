void ff_id3v1_read(AVFormatContext *s)

{

    int ret;

    uint8_t buf[ID3v1_TAG_SIZE];

    int64_t filesize, position = avio_tell(s->pb);



    if (s->pb->seekable) {

        /* XXX: change that */

        filesize = avio_size(s->pb);

        if (filesize > 128) {

            avio_seek(s->pb, filesize - 128, SEEK_SET);

            ret = avio_read(s->pb, buf, ID3v1_TAG_SIZE);

            if (ret == ID3v1_TAG_SIZE) {

                parse_tag(s, buf);

            }

            avio_seek(s->pb, position, SEEK_SET);

        }

    }

}
