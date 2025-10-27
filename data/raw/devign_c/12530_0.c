static void hda_audio_command(HDACodecDevice *hda, uint32_t nid, uint32_t data)

{

    HDAAudioState *a = HDA_AUDIO(hda);

    HDAAudioStream *st;

    const desc_node *node = NULL;

    const desc_param *param;

    uint32_t verb, payload, response, count, shift;



    if ((data & 0x70000) == 0x70000) {

        /* 12/8 id/payload */

        verb = (data >> 8) & 0xfff;

        payload = data & 0x00ff;

    } else {

        /* 4/16 id/payload */

        verb = (data >> 8) & 0xf00;

        payload = data & 0xffff;

    }



    node = hda_codec_find_node(a->desc, nid);

    if (node == NULL) {

        goto fail;

    }

    dprint(a, 2, "%s: nid %d (%s), verb 0x%x, payload 0x%x\n",

           __FUNCTION__, nid, node->name, verb, payload);



    switch (verb) {

    /* all nodes */

    case AC_VERB_PARAMETERS:

        param = hda_codec_find_param(node, payload);

        if (param == NULL) {

            goto fail;

        }

        hda_codec_response(hda, true, param->val);

        break;

    case AC_VERB_GET_SUBSYSTEM_ID:

        hda_codec_response(hda, true, a->desc->iid);

        break;



    /* all functions */

    case AC_VERB_GET_CONNECT_LIST:

        param = hda_codec_find_param(node, AC_PAR_CONNLIST_LEN);

        count = param ? param->val : 0;

        response = 0;

        shift = 0;

        while (payload < count && shift < 32) {

            response |= node->conn[payload] << shift;

            payload++;

            shift += 8;

        }

        hda_codec_response(hda, true, response);

        break;



    /* pin widget */

    case AC_VERB_GET_CONFIG_DEFAULT:

        hda_codec_response(hda, true, node->config);

        break;

    case AC_VERB_GET_PIN_WIDGET_CONTROL:

        hda_codec_response(hda, true, node->pinctl);

        break;

    case AC_VERB_SET_PIN_WIDGET_CONTROL:

        if (node->pinctl != payload) {

            dprint(a, 1, "unhandled pin control bit\n");

        }

        hda_codec_response(hda, true, 0);

        break;



    /* audio in/out widget */

    case AC_VERB_SET_CHANNEL_STREAMID:

        st = a->st + node->stindex;

        if (st->node == NULL) {

            goto fail;

        }

        hda_audio_set_running(st, false);

        st->stream = (payload >> 4) & 0x0f;

        st->channel = payload & 0x0f;

        dprint(a, 2, "%s: stream %d, channel %d\n",

               st->node->name, st->stream, st->channel);

        hda_audio_set_running(st, a->running_real[st->output * 16 + st->stream]);

        hda_codec_response(hda, true, 0);

        break;

    case AC_VERB_GET_CONV:

        st = a->st + node->stindex;

        if (st->node == NULL) {

            goto fail;

        }

        response = st->stream << 4 | st->channel;

        hda_codec_response(hda, true, response);

        break;

    case AC_VERB_SET_STREAM_FORMAT:

        st = a->st + node->stindex;

        if (st->node == NULL) {

            goto fail;

        }

        st->format = payload;

        hda_codec_parse_fmt(st->format, &st->as);

        hda_audio_setup(st);

        hda_codec_response(hda, true, 0);

        break;

    case AC_VERB_GET_STREAM_FORMAT:

        st = a->st + node->stindex;

        if (st->node == NULL) {

            goto fail;

        }

        hda_codec_response(hda, true, st->format);

        break;

    case AC_VERB_GET_AMP_GAIN_MUTE:

        st = a->st + node->stindex;

        if (st->node == NULL) {

            goto fail;

        }

        if (payload & AC_AMP_GET_LEFT) {

            response = st->gain_left | (st->mute_left ? AC_AMP_MUTE : 0);

        } else {

            response = st->gain_right | (st->mute_right ? AC_AMP_MUTE : 0);

        }

        hda_codec_response(hda, true, response);

        break;

    case AC_VERB_SET_AMP_GAIN_MUTE:

        st = a->st + node->stindex;

        if (st->node == NULL) {

            goto fail;

        }

        dprint(a, 1, "amp (%s): %s%s%s%s index %d  gain %3d %s\n",

               st->node->name,

               (payload & AC_AMP_SET_OUTPUT) ? "o" : "-",

               (payload & AC_AMP_SET_INPUT)  ? "i" : "-",

               (payload & AC_AMP_SET_LEFT)   ? "l" : "-",

               (payload & AC_AMP_SET_RIGHT)  ? "r" : "-",

               (payload & AC_AMP_SET_INDEX) >> AC_AMP_SET_INDEX_SHIFT,

               (payload & AC_AMP_GAIN),

               (payload & AC_AMP_MUTE) ? "muted" : "");

        if (payload & AC_AMP_SET_LEFT) {

            st->gain_left = payload & AC_AMP_GAIN;

            st->mute_left = payload & AC_AMP_MUTE;

        }

        if (payload & AC_AMP_SET_RIGHT) {

            st->gain_right = payload & AC_AMP_GAIN;

            st->mute_right = payload & AC_AMP_MUTE;

        }

        hda_audio_set_amp(st);

        hda_codec_response(hda, true, 0);

        break;



    /* not supported */

    case AC_VERB_SET_POWER_STATE:

    case AC_VERB_GET_POWER_STATE:

    case AC_VERB_GET_SDI_SELECT:

        hda_codec_response(hda, true, 0);

        break;

    default:

        goto fail;

    }

    return;



fail:

    dprint(a, 1, "%s: not handled: nid %d (%s), verb 0x%x, payload 0x%x\n",

           __FUNCTION__, nid, node ? node->name : "?", verb, payload);

    hda_codec_response(hda, true, 0);

}
