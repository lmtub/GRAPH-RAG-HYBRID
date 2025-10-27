int ff_init_vlc_sparse(VLC *vlc, int nb_bits, int nb_codes,

             const void *bits, int bits_wrap, int bits_size,

             const void *codes, int codes_wrap, int codes_size,

             const void *symbols, int symbols_wrap, int symbols_size,

             int flags)

{

    VLCcode *buf;

    int i, j, ret;



    vlc->bits = nb_bits;

    if(flags & INIT_VLC_USE_NEW_STATIC){

        VLC dyn_vlc = *vlc;



        if (vlc->table_size)

            return 0;



        ret = ff_init_vlc_sparse(&dyn_vlc, nb_bits, nb_codes,

                                 bits, bits_wrap, bits_size,

                                 codes, codes_wrap, codes_size,

                                 symbols, symbols_wrap, symbols_size,

                                 flags & ~INIT_VLC_USE_NEW_STATIC);

        av_assert0(ret >= 0);

        av_assert0(dyn_vlc.table_size <= vlc->table_allocated);

        if(dyn_vlc.table_size < vlc->table_allocated)

            av_log(NULL, AV_LOG_ERROR, "needed %d had %d\n", dyn_vlc.table_size, vlc->table_allocated);

        memcpy(vlc->table, dyn_vlc.table, dyn_vlc.table_size * sizeof(*vlc->table));

        vlc->table_size = dyn_vlc.table_size;

        ff_free_vlc(&dyn_vlc);

        return 0;

    }else {

        vlc->table = NULL;

        vlc->table_allocated = 0;

        vlc->table_size = 0;

    }



    av_dlog(NULL, "build table nb_codes=%d\n", nb_codes);



    buf = av_malloc((nb_codes+1)*sizeof(VLCcode));



    av_assert0(symbols_size <= 2 || !symbols);

    j = 0;

#define COPY(condition)\

    for (i = 0; i < nb_codes; i++) {\

        GET_DATA(buf[j].bits, bits, i, bits_wrap, bits_size);\

        if (!(condition))\

            continue;\

        if (buf[j].bits > 3*nb_bits || buf[j].bits>32) {\

            av_log(NULL, AV_LOG_ERROR, "Too long VLC in init_vlc\n");\


            return -1;\

        }\

        GET_DATA(buf[j].code, codes, i, codes_wrap, codes_size);\

        if (buf[j].code >= (1LL<<buf[j].bits)) {\

            av_log(NULL, AV_LOG_ERROR, "Invalid code in init_vlc\n");\


            return -1;\

        }\

        if (flags & INIT_VLC_LE)\

            buf[j].code = bitswap_32(buf[j].code);\

        else\

            buf[j].code <<= 32 - buf[j].bits;\

        if (symbols)\

            GET_DATA(buf[j].symbol, symbols, i, symbols_wrap, symbols_size)\

        else\

            buf[j].symbol = i;\

        j++;\

    }

    COPY(buf[j].bits > nb_bits);

    // qsort is the slowest part of init_vlc, and could probably be improved or avoided

    qsort(buf, j, sizeof(VLCcode), compare_vlcspec);

    COPY(buf[j].bits && buf[j].bits <= nb_bits);

    nb_codes = j;



    ret = build_table(vlc, nb_bits, nb_codes, buf, flags);



    av_free(buf);

    if (ret < 0) {

        av_freep(&vlc->table);

        return -1;

    }

    return 0;

}