static int mov_read_chpl(MOVContext *c, AVIOContext *pb, MOVAtom atom)

{

    int64_t start;

    int i, nb_chapters, str_len, version;

    char str[256+1];



    if ((atom.size -= 5) < 0)

        return 0;



    version = avio_r8(pb);

    avio_rb24(pb);

    if (version)

        avio_rb32(pb); // ???

    nb_chapters = avio_r8(pb);



    for (i = 0; i < nb_chapters; i++) {

        if (atom.size < 9)

            return 0;



        start = avio_rb64(pb);

        str_len = avio_r8(pb);



        if ((atom.size -= 9+str_len) < 0)

            return 0;



        avio_read(pb, str, str_len);

        str[str_len] = 0;

        avpriv_new_chapter(c->fc, i, (AVRational){1,10000000}, start, AV_NOPTS_VALUE, str);

    }

    return 0;

}
