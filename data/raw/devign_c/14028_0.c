static av_always_inline int encode_line(FFV1Context *s, int w,

                                        int16_t *sample[3],

                                        int plane_index, int bits)

{

    PlaneContext *const p = &s->plane[plane_index];

    RangeCoder *const c   = &s->c;

    int x;

    int run_index = s->run_index;

    int run_count = 0;

    int run_mode  = 0;



    if (s->ac) {

        if (c->bytestream_end - c->bytestream < w * 20) {

            av_log(s->avctx, AV_LOG_ERROR, "encoded frame too large\n");

            return AVERROR_INVALIDDATA;

        }

    } else {

        if (s->pb.buf_end - s->pb.buf - (put_bits_count(&s->pb) >> 3) < w * 4) {

            av_log(s->avctx, AV_LOG_ERROR, "encoded frame too large\n");

            return AVERROR_INVALIDDATA;

        }

    }



    for (x = 0; x < w; x++) {

        int diff, context;



        context = get_context(p, sample[0] + x, sample[1] + x, sample[2] + x);

        diff    = sample[0][x] - predict(sample[0] + x, sample[1] + x);



        if (context < 0) {

            context = -context;

            diff    = -diff;

        }



        diff = fold(diff, bits);



        if (s->ac) {

            if (s->flags & CODEC_FLAG_PASS1) {

                put_symbol_inline(c, p->state[context], diff, 1, s->rc_stat,

                                  s->rc_stat2[p->quant_table_index][context]);

            } else {

                put_symbol_inline(c, p->state[context], diff, 1, NULL, NULL);

            }

        } else {

            if (context == 0)

                run_mode = 1;



            if (run_mode) {

                if (diff) {

                    while (run_count >= 1 << ff_log2_run[run_index]) {

                        run_count -= 1 << ff_log2_run[run_index];

                        run_index++;

                        put_bits(&s->pb, 1, 1);

                    }



                    put_bits(&s->pb, 1 + ff_log2_run[run_index], run_count);

                    if (run_index)

                        run_index--;

                    run_count = 0;

                    run_mode  = 0;

                    if (diff > 0)

                        diff--;

                } else {

                    run_count++;

                }

            }



            av_dlog(s->avctx, "count:%d index:%d, mode:%d, x:%d pos:%d\n",

                    run_count, run_index, run_mode, x,

                    (int)put_bits_count(&s->pb));



            if (run_mode == 0)

                put_vlc_symbol(&s->pb, &p->vlc_state[context], diff, bits);

        }

    }

    if (run_mode) {

        while (run_count >= 1 << ff_log2_run[run_index]) {

            run_count -= 1 << ff_log2_run[run_index];

            run_index++;

            put_bits(&s->pb, 1, 1);

        }



        if (run_count)

            put_bits(&s->pb, 1, 1);

    }

    s->run_index = run_index;



    return 0;

}
