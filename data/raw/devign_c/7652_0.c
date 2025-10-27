static int metadata_parse(FLACContext *s)

{

    int i, metadata_last, metadata_type, metadata_size;

    int initial_pos= get_bits_count(&s->gb);



    if (show_bits_long(&s->gb, 32) == MKBETAG('f','L','a','C')) {

        skip_bits_long(&s->gb, 32);



        do {

            metadata_last = get_bits1(&s->gb);

            metadata_type = get_bits(&s->gb, 7);

            metadata_size = get_bits_long(&s->gb, 24);



            if (get_bits_count(&s->gb) + 8*metadata_size > s->gb.size_in_bits) {

                skip_bits_long(&s->gb, initial_pos - get_bits_count(&s->gb));

                break;

            }



            if (metadata_size) {

                switch (metadata_type) {

                case FLAC_METADATA_TYPE_STREAMINFO:

                    if (!s->got_streaminfo) {

                        ff_flac_parse_streaminfo(s->avctx, (FLACStreaminfo *)s,

                                                 s->gb.buffer+get_bits_count(&s->gb)/8);

                        allocate_buffers(s);

                        s->got_streaminfo = 1;

                    }

                default:

                    for (i = 0; i < metadata_size; i++)

                        skip_bits(&s->gb, 8);

                }

            }

        } while (!metadata_last);



        return 1;

    }

    return 0;

}
