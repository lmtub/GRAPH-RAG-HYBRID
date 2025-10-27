static int mjpeg_decode_dht(MJpegDecodeContext *s)

{

    int len, index, i, class, n, v, code_max;

    uint8_t bits_table[17];

    uint8_t val_table[256];

    

    len = get_bits(&s->gb, 16) - 2;



    while (len > 0) {

        if (len < 17)

            return -1;

        class = get_bits(&s->gb, 4);

        if (class >= 2)

            return -1;

        index = get_bits(&s->gb, 4);

        if (index >= 4)

            return -1;

        n = 0;

        for(i=1;i<=16;i++) {

            bits_table[i] = get_bits(&s->gb, 8);

            n += bits_table[i];

        }

        len -= 17;

        if (len < n || n > 256)

            return -1;



        code_max = 0;

        for(i=0;i<n;i++) {

            v = get_bits(&s->gb, 8);

            if (v > code_max)

                code_max = v;

            val_table[i] = v;

        }

        len -= n;



        /* build VLC and flush previous vlc if present */

        free_vlc(&s->vlcs[class][index]);

        dprintf("class=%d index=%d nb_codes=%d\n",

               class, index, code_max + 1);

        if(build_vlc(&s->vlcs[class][index], bits_table, val_table, code_max + 1) < 0){

            return -1;

        }

    }

    return 0;

}
