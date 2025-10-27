static int mov_read_hdlr(MOVContext *c, ByteIOContext *pb, MOV_atom_t atom)

{

    AVStream *st = c->fc->streams[c->fc->nb_streams-1];

    int len = 0;

    uint8_t *buf;

    uint32_t type;

    uint32_t ctype;



    print_atom("hdlr", atom);



    get_byte(pb); /* version */

    get_byte(pb); get_byte(pb); get_byte(pb); /* flags */



    /* component type */

    ctype = get_le32(pb);

    type = get_le32(pb); /* component subtype */



#ifdef DEBUG

    printf("ctype= %c%c%c%c (0x%08lx)\n", *((char *)&ctype), ((char *)&ctype)[1], ((char *)&ctype)[2], ((char *)&ctype)[3], (long) ctype);

    printf("stype= %c%c%c%c\n", *((char *)&type), ((char *)&type)[1], ((char *)&type)[2], ((char *)&type)[3]);

#endif

#ifdef DEBUG

/* XXX: yeah this is ugly... */

    if(ctype == MKTAG('m', 'h', 'l', 'r')) { /* MOV */

        if(type == MKTAG('v', 'i', 'd', 'e'))

            puts("hdlr: vide");

        else if(type == MKTAG('s', 'o', 'u', 'n'))

            puts("hdlr: soun");

    } else if(ctype == 0) { /* MP4 */

        if(type == MKTAG('v', 'i', 'd', 'e'))

            puts("hdlr: vide");

        else if(type == MKTAG('s', 'o', 'u', 'n'))

            puts("hdlr: soun");

        else if(type == MKTAG('o', 'd', 's', 'm'))

            puts("hdlr: odsm");

        else if(type == MKTAG('s', 'd', 's', 'm'))

            puts("hdlr: sdsm");

    } else puts("hdlr: meta");

#endif



    if(ctype == MKTAG('m', 'h', 'l', 'r')) { /* MOV */

        /* helps parsing the string hereafter... */

        c->mp4 = 0;

        if(type == MKTAG('v', 'i', 'd', 'e'))

            st->codec.codec_type = CODEC_TYPE_VIDEO;

        else if(type == MKTAG('s', 'o', 'u', 'n'))

            st->codec.codec_type = CODEC_TYPE_AUDIO;

    } else if(ctype == 0) { /* MP4 */

        /* helps parsing the string hereafter... */

        c->mp4 = 1;

        if(type == MKTAG('v', 'i', 'd', 'e'))

            st->codec.codec_type = CODEC_TYPE_VIDEO;

        else if(type == MKTAG('s', 'o', 'u', 'n'))

            st->codec.codec_type = CODEC_TYPE_AUDIO;

    }

    get_be32(pb); /* component  manufacture */

    get_be32(pb); /* component flags */

    get_be32(pb); /* component flags mask */



    if(atom.size <= 24)

        return 0; /* nothing left to read */

    /* XXX: MP4 uses a C string, not a pascal one */

    /* component name */



    if(c->mp4) {

        /* .mp4: C string */

        while(get_byte(pb) && (++len < (atom.size - 24)));

    } else {

        /* .mov: PASCAL string */

        len = get_byte(pb);

#ifdef DEBUG

	buf = (uint8_t*) av_malloc(len+1);

	if (buf) {

	    get_buffer(pb, buf, len);

	    buf[len] = '\0';

	    printf("**buf='%s'\n", buf);

	    av_free(buf);

	} else

#endif

	    url_fskip(pb, len);

    }



    return 0;

}
