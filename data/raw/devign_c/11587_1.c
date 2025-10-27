static int gif_read_extension(GifState *s)

{

    ByteIOContext *f = s->f;

    int ext_code, ext_len, i, gce_flags, gce_transparent_index;



    /* extension */

    ext_code = get_byte(f);

    ext_len = get_byte(f);

#ifdef DEBUG

    printf("gif: ext_code=0x%x len=%d\n", ext_code, ext_len);

#endif

    switch(ext_code) {

    case 0xf9:

        if (ext_len != 4)

            goto discard_ext;

        s->transparent_color_index = -1;

        gce_flags = get_byte(f);

        s->gce_delay = get_le16(f);

        gce_transparent_index = get_byte(f);

        if (gce_flags & 0x01)

            s->transparent_color_index = gce_transparent_index;

        else

            s->transparent_color_index = -1;

        s->gce_disposal = (gce_flags >> 2) & 0x7;

#ifdef DEBUG

        printf("gif: gce_flags=%x delay=%d tcolor=%d disposal=%d\n",

               gce_flags, s->gce_delay,

               s->transparent_color_index, s->gce_disposal);

#endif

        ext_len = get_byte(f);

        break;

    }



    /* NOTE: many extension blocks can come after */

 discard_ext:

    while (ext_len != 0) {

        for (i = 0; i < ext_len; i++)

            get_byte(f);

        ext_len = get_byte(f);

#ifdef DEBUG

        printf("gif: ext_len1=%d\n", ext_len);

#endif

    }

    return 0;

}
