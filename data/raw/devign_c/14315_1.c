int ff_mov_read_esds(AVFormatContext *fc, AVIOContext *pb, MOVAtom atom)

{

    AVStream *st;

    int tag;



    if (fc->nb_streams < 1)

        return 0;

    st = fc->streams[fc->nb_streams-1];



    avio_rb32(pb); /* version + flags */

    ff_mp4_read_descr(fc, pb, &tag);

    if (tag == MP4ESDescrTag) {

        ff_mp4_parse_es_descr(pb, NULL);

    } else

        avio_rb16(pb); /* ID */



    ff_mp4_read_descr(fc, pb, &tag);

    if (tag == MP4DecConfigDescrTag)

        ff_mp4_read_dec_config_descr(fc, st, pb);

    return 0;

}
