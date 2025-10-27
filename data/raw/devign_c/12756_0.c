static int hda_audio_init(HDACodecDevice *hda, const struct desc_codec *desc)

{

    HDAAudioState *a = HDA_AUDIO(hda);

    HDAAudioStream *st;

    const desc_node *node;

    const desc_param *param;

    uint32_t i, type;



    a->desc = desc;

    a->name = object_get_typename(OBJECT(a));

    dprint(a, 1, "%s: cad %d\n", __FUNCTION__, a->hda.cad);



    AUD_register_card("hda", &a->card);

    for (i = 0; i < a->desc->nnodes; i++) {

        node = a->desc->nodes + i;

        param = hda_codec_find_param(node, AC_PAR_AUDIO_WIDGET_CAP);

        if (param == NULL) {

            continue;

        }

        type = (param->val & AC_WCAP_TYPE) >> AC_WCAP_TYPE_SHIFT;

        switch (type) {

        case AC_WID_AUD_OUT:

        case AC_WID_AUD_IN:

            assert(node->stindex < ARRAY_SIZE(a->st));

            st = a->st + node->stindex;

            st->state = a;

            st->node = node;

            if (type == AC_WID_AUD_OUT) {

                /* unmute output by default */

                st->gain_left = QEMU_HDA_AMP_STEPS;

                st->gain_right = QEMU_HDA_AMP_STEPS;

                st->bpos = sizeof(st->buf);

                st->output = true;

            } else {

                st->output = false;

            }

            st->format = AC_FMT_TYPE_PCM | AC_FMT_BITS_16 |

                (1 << AC_FMT_CHAN_SHIFT);

            hda_codec_parse_fmt(st->format, &st->as);

            hda_audio_setup(st);

            break;

        }

    }

    return 0;

}
