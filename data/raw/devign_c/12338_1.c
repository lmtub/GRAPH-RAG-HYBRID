static int parse_audio(DBEContext *s, int start, int end, int seg_id)

{

    int ch, ret, key = parse_key(s);



    for (ch = start; ch < end; ch++) {

        if (!s->ch_size[ch]) {

            s->channels[seg_id][ch].nb_groups = 0;

            continue;

        }

        if ((ret = convert_input(s, s->ch_size[ch], key)) < 0)

            return ret;

        if ((ret = parse_channel(s, ch, seg_id)) < 0) {

            if (s->avctx->err_recognition & AV_EF_EXPLODE)

                return ret;

            s->channels[seg_id][ch].nb_groups = 0;

        }

        skip_input(s, s->ch_size[ch]);

    }



    skip_input(s, 1);

    return 0;

}
