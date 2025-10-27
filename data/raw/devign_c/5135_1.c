int ff_ps_read_data(AVCodecContext *avctx, GetBitContext *gb_host, PSContext *ps, int bits_left)

{

    int e;

    int bit_count_start = get_bits_count(gb_host);

    int header;

    int bits_consumed;

    GetBitContext gbc = *gb_host, *gb = &gbc;



    header = get_bits1(gb);

    if (header) {     //enable_ps_header

        ps->enable_iid = get_bits1(gb);

        if (ps->enable_iid) {

            int iid_mode = get_bits(gb, 3);

            if (iid_mode > 5) {

                av_log(avctx, AV_LOG_ERROR, "iid_mode %d is reserved.\n",

                       iid_mode);

                goto err;

            }

            ps->nr_iid_par    = nr_iidicc_par_tab[iid_mode];

            ps->iid_quant     = iid_mode > 2;

            ps->nr_ipdopd_par = nr_iidopd_par_tab[iid_mode];

        }

        ps->enable_icc = get_bits1(gb);

        if (ps->enable_icc) {

            ps->icc_mode = get_bits(gb, 3);

            if (ps->icc_mode > 5) {

                av_log(avctx, AV_LOG_ERROR, "icc_mode %d is reserved.\n",

                       ps->icc_mode);

                goto err;

            }

            ps->nr_icc_par = nr_iidicc_par_tab[ps->icc_mode];

        }

        ps->enable_ext = get_bits1(gb);

    }



    ps->frame_class = get_bits1(gb);

    ps->num_env_old = ps->num_env;

    ps->num_env     = num_env_tab[ps->frame_class][get_bits(gb, 2)];



    ps->border_position[0] = -1;

    if (ps->frame_class) {

        for (e = 1; e <= ps->num_env; e++)

            ps->border_position[e] = get_bits(gb, 5);

    } else

        for (e = 1; e <= ps->num_env; e++)

            ps->border_position[e] = (e * numQMFSlots >> ff_log2_tab[ps->num_env]) - 1;



    if (ps->enable_iid) {

        for (e = 0; e < ps->num_env; e++) {

            int dt = get_bits1(gb);

            if (read_iid_data(avctx, gb, ps, ps->iid_par, huff_iid[2*dt+ps->iid_quant], e, dt))

                goto err;

        }

    } else




    if (ps->enable_icc)

        for (e = 0; e < ps->num_env; e++) {

            int dt = get_bits1(gb);

            if (read_icc_data(avctx, gb, ps, ps->icc_par, dt ? huff_icc_dt : huff_icc_df, e, dt))

                goto err;

        }

    else




    if (ps->enable_ext) {

        int cnt = get_bits(gb, 4);

        if (cnt == 15) {

            cnt += get_bits(gb, 8);

        }

        cnt *= 8;

        while (cnt > 7) {

            int ps_extension_id = get_bits(gb, 2);

            cnt -= 2 + ps_read_extension_data(gb, ps, ps_extension_id);

        }

        if (cnt < 0) {

            av_log(avctx, AV_LOG_ERROR, "ps extension overflow %d\n", cnt);

            goto err;

        }

        skip_bits(gb, cnt);

    }



    ps->enable_ipdopd &= !PS_BASELINE;



    //Fix up envelopes

    if (!ps->num_env || ps->border_position[ps->num_env] < numQMFSlots - 1) {

        //Create a fake envelope

        int source = ps->num_env ? ps->num_env - 1 : ps->num_env_old - 1;

        if (source >= 0 && source != ps->num_env) {

            if (ps->enable_iid) {

                memcpy(ps->iid_par+ps->num_env, ps->iid_par+source, sizeof(ps->iid_par[0]));

            }

            if (ps->enable_icc) {

                memcpy(ps->icc_par+ps->num_env, ps->icc_par+source, sizeof(ps->icc_par[0]));

            }

            if (ps->enable_ipdopd) {

                memcpy(ps->ipd_par+ps->num_env, ps->ipd_par+source, sizeof(ps->ipd_par[0]));

                memcpy(ps->opd_par+ps->num_env, ps->opd_par+source, sizeof(ps->opd_par[0]));

            }

        }

        ps->num_env++;

        ps->border_position[ps->num_env] = numQMFSlots - 1;

    }





    ps->is34bands_old = ps->is34bands;

    if (!PS_BASELINE && (ps->enable_iid || ps->enable_icc))

        ps->is34bands = (ps->enable_iid && ps->nr_iid_par == 34) ||

                        (ps->enable_icc && ps->nr_icc_par == 34);



    //Baseline

    if (!ps->enable_ipdopd) {



    }



    if (header)

        ps->start = 1;



    bits_consumed = get_bits_count(gb) - bit_count_start;

    if (bits_consumed <= bits_left) {

        skip_bits_long(gb_host, bits_consumed);

        return bits_consumed;

    }

    av_log(avctx, AV_LOG_ERROR, "Expected to read %d PS bits actually read %d.\n", bits_left, bits_consumed);

err:

    ps->start = 0;

    skip_bits_long(gb_host, bits_left);





    return bits_left;

}