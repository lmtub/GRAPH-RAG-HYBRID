static int vorbis_parse_setup_hdr_codebooks(vorbis_context *vc)

{

    unsigned cb;

    uint8_t  *tmp_vlc_bits;

    uint32_t *tmp_vlc_codes;

    GetBitContext *gb = &vc->gb;

    uint16_t *codebook_multiplicands;

    int ret = 0;



    vc->codebook_count = get_bits(gb, 8) + 1;



    av_dlog(NULL, " Codebooks: %d \n", vc->codebook_count);



    vc->codebooks = av_mallocz(vc->codebook_count * sizeof(*vc->codebooks));

    tmp_vlc_bits  = av_mallocz(V_MAX_VLCS * sizeof(*tmp_vlc_bits));

    tmp_vlc_codes = av_mallocz(V_MAX_VLCS * sizeof(*tmp_vlc_codes));

    codebook_multiplicands = av_malloc(V_MAX_VLCS * sizeof(*codebook_multiplicands));



    for (cb = 0; cb < vc->codebook_count; ++cb) {

        vorbis_codebook *codebook_setup = &vc->codebooks[cb];

        unsigned ordered, t, entries, used_entries = 0;



        av_dlog(NULL, " %u. Codebook\n", cb);



        if (get_bits(gb, 24) != 0x564342) {

            av_log(vc->avctx, AV_LOG_ERROR,

                   " %u. Codebook setup data corrupt.\n", cb);

            ret = AVERROR_INVALIDDATA;

            goto error;

        }



        codebook_setup->dimensions=get_bits(gb, 16);

        if (codebook_setup->dimensions > 16 || codebook_setup->dimensions == 0) {

            av_log(vc->avctx, AV_LOG_ERROR,

                   " %u. Codebook's dimension is invalid (%d).\n",

                   cb, codebook_setup->dimensions);

            ret = AVERROR_INVALIDDATA;

            goto error;

        }

        entries = get_bits(gb, 24);

        if (entries > V_MAX_VLCS) {

            av_log(vc->avctx, AV_LOG_ERROR,

                   " %u. Codebook has too many entries (%u).\n",

                   cb, entries);

            ret = AVERROR_INVALIDDATA;

            goto error;

        }



        ordered = get_bits1(gb);



        av_dlog(NULL, " codebook_dimensions %d, codebook_entries %u\n",

                codebook_setup->dimensions, entries);



        if (!ordered) {

            unsigned ce, flag;

            unsigned sparse = get_bits1(gb);



            av_dlog(NULL, " not ordered \n");



            if (sparse) {

                av_dlog(NULL, " sparse \n");



                used_entries = 0;

                for (ce = 0; ce < entries; ++ce) {

                    flag = get_bits1(gb);

                    if (flag) {

                        tmp_vlc_bits[ce] = get_bits(gb, 5) + 1;

                        ++used_entries;

                    } else

                        tmp_vlc_bits[ce] = 0;

                }

            } else {

                av_dlog(NULL, " not sparse \n");



                used_entries = entries;

                for (ce = 0; ce < entries; ++ce)

                    tmp_vlc_bits[ce] = get_bits(gb, 5) + 1;

            }

        } else {

            unsigned current_entry  = 0;

            unsigned current_length = get_bits(gb, 5) + 1;



            av_dlog(NULL, " ordered, current length: %u\n", current_length);  //FIXME



            used_entries = entries;

            for (; current_entry < used_entries && current_length <= 32; ++current_length) {

                unsigned i, number;



                av_dlog(NULL, " number bits: %u ", ilog(entries - current_entry));



                number = get_bits(gb, ilog(entries - current_entry));



                av_dlog(NULL, " number: %u\n", number);



                for (i = current_entry; i < number+current_entry; ++i)

                    if (i < used_entries)

                        tmp_vlc_bits[i] = current_length;



                current_entry+=number;

            }

            if (current_entry>used_entries) {

                av_log(vc->avctx, AV_LOG_ERROR, " More codelengths than codes in codebook. \n");

                ret = AVERROR_INVALIDDATA;

                goto error;

            }

        }



        codebook_setup->lookup_type = get_bits(gb, 4);



        av_dlog(NULL, " lookup type: %d : %s \n", codebook_setup->lookup_type,

                codebook_setup->lookup_type ? "vq" : "no lookup");



// If the codebook is used for (inverse) VQ, calculate codevectors.



        if (codebook_setup->lookup_type == 1) {

            unsigned i, j, k;

            unsigned codebook_lookup_values = ff_vorbis_nth_root(entries, codebook_setup->dimensions);



            float codebook_minimum_value = vorbisfloat2float(get_bits_long(gb, 32));

            float codebook_delta_value   = vorbisfloat2float(get_bits_long(gb, 32));

            unsigned codebook_value_bits = get_bits(gb, 4) + 1;

            unsigned codebook_sequence_p = get_bits1(gb);



            av_dlog(NULL, " We expect %d numbers for building the codevectors. \n",

                    codebook_lookup_values);

            av_dlog(NULL, "  delta %f minmum %f \n",

                    codebook_delta_value, codebook_minimum_value);



            for (i = 0; i < codebook_lookup_values; ++i) {

                codebook_multiplicands[i] = get_bits(gb, codebook_value_bits);



                av_dlog(NULL, " multiplicands*delta+minmum : %e \n",

                        (float)codebook_multiplicands[i] * codebook_delta_value + codebook_minimum_value);

                av_dlog(NULL, " multiplicand %u\n", codebook_multiplicands[i]);

            }



// Weed out unused vlcs and build codevector vector

            codebook_setup->codevectors = used_entries ? av_mallocz(used_entries *

                                                                    codebook_setup->dimensions *

                                                                    sizeof(*codebook_setup->codevectors))

                                                       : NULL;

            for (j = 0, i = 0; i < entries; ++i) {

                unsigned dim = codebook_setup->dimensions;



                if (tmp_vlc_bits[i]) {

                    float last = 0.0;

                    unsigned lookup_offset = i;



                    av_dlog(vc->avctx, "Lookup offset %u ,", i);



                    for (k = 0; k < dim; ++k) {

                        unsigned multiplicand_offset = lookup_offset % codebook_lookup_values;

                        codebook_setup->codevectors[j * dim + k] = codebook_multiplicands[multiplicand_offset] * codebook_delta_value + codebook_minimum_value + last;

                        if (codebook_sequence_p)

                            last = codebook_setup->codevectors[j * dim + k];

                        lookup_offset/=codebook_lookup_values;

                    }

                    tmp_vlc_bits[j] = tmp_vlc_bits[i];



                    av_dlog(vc->avctx, "real lookup offset %u, vector: ", j);

                    for (k = 0; k < dim; ++k)

                        av_dlog(vc->avctx, " %f ",

                                codebook_setup->codevectors[j * dim + k]);

                    av_dlog(vc->avctx, "\n");



                    ++j;

                }

            }

            if (j != used_entries) {

                av_log(vc->avctx, AV_LOG_ERROR, "Bug in codevector vector building code. \n");

                ret = AVERROR_INVALIDDATA;

                goto error;

            }

            entries = used_entries;

        } else if (codebook_setup->lookup_type >= 2) {

            av_log(vc->avctx, AV_LOG_ERROR, "Codebook lookup type not supported. \n");

            ret = AVERROR_INVALIDDATA;

            goto error;

        }



// Initialize VLC table

        if (ff_vorbis_len2vlc(tmp_vlc_bits, tmp_vlc_codes, entries)) {

            av_log(vc->avctx, AV_LOG_ERROR, " Invalid code lengths while generating vlcs. \n");

            ret = AVERROR_INVALIDDATA;

            goto error;

        }

        codebook_setup->maxdepth = 0;

        for (t = 0; t < entries; ++t)

            if (tmp_vlc_bits[t] >= codebook_setup->maxdepth)

                codebook_setup->maxdepth = tmp_vlc_bits[t];



        if (codebook_setup->maxdepth > 3 * V_NB_BITS)

            codebook_setup->nb_bits = V_NB_BITS2;

        else

            codebook_setup->nb_bits = V_NB_BITS;



        codebook_setup->maxdepth = (codebook_setup->maxdepth+codebook_setup->nb_bits - 1) / codebook_setup->nb_bits;



        if ((ret = init_vlc(&codebook_setup->vlc, codebook_setup->nb_bits,

                            entries, tmp_vlc_bits, sizeof(*tmp_vlc_bits),

                            sizeof(*tmp_vlc_bits), tmp_vlc_codes,

                            sizeof(*tmp_vlc_codes), sizeof(*tmp_vlc_codes),

                            INIT_VLC_LE))) {

            av_log(vc->avctx, AV_LOG_ERROR, " Error generating vlc tables. \n");

            goto error;

        }

    }



    av_free(tmp_vlc_bits);

    av_free(tmp_vlc_codes);

    av_free(codebook_multiplicands);

    return 0;



// Error:

error:

    av_free(tmp_vlc_bits);

    av_free(tmp_vlc_codes);

    av_free(codebook_multiplicands);

    return ret;

}
