static int build_table(VLC *vlc, int table_nb_bits, int nb_codes,

                       VLCcode *codes, int flags)

{

    int table_size, table_index, index, code_prefix, symbol, subtable_bits;

    int i, j, k, n, nb, inc;

    uint32_t code;

    VLC_TYPE (*table)[2];



    table_size = 1 << table_nb_bits;

    if (table_nb_bits > 30)

       return -1;

    table_index = alloc_table(vlc, table_size, flags & INIT_VLC_USE_NEW_STATIC);

    av_dlog(NULL, "new table index=%d size=%d\n", table_index, table_size);

    if (table_index < 0)

        return table_index;

    table = &vlc->table[table_index];



    for (i = 0; i < table_size; i++) {

        table[i][1] = 0; //bits

        table[i][0] = -1; //codes

    }



    /* first pass: map codes and compute auxiliary table sizes */

    for (i = 0; i < nb_codes; i++) {

        n      = codes[i].bits;

        code   = codes[i].code;

        symbol = codes[i].symbol;

        av_dlog(NULL, "i=%d n=%d code=0x%x\n", i, n, code);

        if (n <= table_nb_bits) {

            /* no need to add another table */

            j = code >> (32 - table_nb_bits);

            nb = 1 << (table_nb_bits - n);

            inc = 1;

            if (flags & INIT_VLC_LE) {

                j = bitswap_32(code);

                inc = 1 << n;

            }

            for (k = 0; k < nb; k++) {

                av_dlog(NULL, "%4x: code=%d n=%d\n", j, i, n);

                if (table[j][1] /*bits*/ != 0) {

                    av_log(NULL, AV_LOG_ERROR, "incorrect codes\n");

                    return AVERROR_INVALIDDATA;

                }

                table[j][1] = n; //bits

                table[j][0] = symbol;

                j += inc;

            }

        } else {

            /* fill auxiliary table recursively */

            n -= table_nb_bits;

            code_prefix = code >> (32 - table_nb_bits);

            subtable_bits = n;

            codes[i].bits = n;

            codes[i].code = code << table_nb_bits;

            for (k = i+1; k < nb_codes; k++) {

                n = codes[k].bits - table_nb_bits;

                if (n <= 0)

                    break;

                code = codes[k].code;

                if (code >> (32 - table_nb_bits) != code_prefix)

                    break;

                codes[k].bits = n;

                codes[k].code = code << table_nb_bits;

                subtable_bits = FFMAX(subtable_bits, n);

            }

            subtable_bits = FFMIN(subtable_bits, table_nb_bits);

            j = (flags & INIT_VLC_LE) ? bitswap_32(code_prefix) >> (32 - table_nb_bits) : code_prefix;

            table[j][1] = -subtable_bits;

            av_dlog(NULL, "%4x: n=%d (subtable)\n",

                    j, codes[i].bits + table_nb_bits);

            index = build_table(vlc, subtable_bits, k-i, codes+i, flags);

            if (index < 0)

                return index;

            /* note: realloc has been done, so reload tables */

            table = &vlc->table[table_index];

            table[j][0] = index; //code

            av_assert0(table[j][0] == index);

            i = k-1;

        }

    }

    return table_index;

}
