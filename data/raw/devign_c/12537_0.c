static void l2cap_command(struct l2cap_instance_s *l2cap, int code, int id,

                const uint8_t *params, int len)

{

    int err;



#if 0

    /* TODO: do the IDs really have to be in sequence?  */

    if (!id || (id != l2cap->last_id && id != l2cap->next_id)) {

        fprintf(stderr, "%s: out of sequence command packet ignored.\n",

                        __func__);

        return;

    }

#else

    l2cap->next_id = id;

#endif

    if (id == l2cap->next_id) {

        l2cap->last_id = l2cap->next_id;

        l2cap->next_id = l2cap->next_id == 255 ? 1 : l2cap->next_id + 1;

    } else {

        /* TODO: Need to re-send the same response, without re-executing

         * the corresponding command!  */

    }



    switch (code) {

    case L2CAP_COMMAND_REJ:

        if (unlikely(len != 2 && len != 4 && len != 6)) {

            err = L2CAP_REJ_CMD_NOT_UNDERSTOOD;

            goto reject;

        }



        /* We never issue commands other than Command Reject currently.  */

        fprintf(stderr, "%s: stray Command Reject (%02x, %04x) "

                        "packet, ignoring.\n", __func__, id,

                        le16_to_cpu(((l2cap_cmd_rej *) params)->reason));

        break;



    case L2CAP_CONN_REQ:

        if (unlikely(len != L2CAP_CONN_REQ_SIZE)) {

            err = L2CAP_REJ_CMD_NOT_UNDERSTOOD;

            goto reject;

        }



        l2cap_channel_open_req_msg(l2cap,

                        le16_to_cpu(((l2cap_conn_req *) params)->psm),

                        le16_to_cpu(((l2cap_conn_req *) params)->scid));

        break;



    case L2CAP_CONN_RSP:

        if (unlikely(len != L2CAP_CONN_RSP_SIZE)) {

            err = L2CAP_REJ_CMD_NOT_UNDERSTOOD;

            goto reject;

        }



        /* We never issue Connection Requests currently. TODO  */

        fprintf(stderr, "%s: unexpected Connection Response (%02x) "

                        "packet, ignoring.\n", __func__, id);

        break;



    case L2CAP_CONF_REQ:

        if (unlikely(len < L2CAP_CONF_REQ_SIZE(0))) {

            err = L2CAP_REJ_CMD_NOT_UNDERSTOOD;

            goto reject;

        }



        l2cap_channel_config_req_msg(l2cap,

                        le16_to_cpu(((l2cap_conf_req *) params)->flags) & 1,

                        le16_to_cpu(((l2cap_conf_req *) params)->dcid),

                        ((l2cap_conf_req *) params)->data,

                        len - L2CAP_CONF_REQ_SIZE(0));

        break;



    case L2CAP_CONF_RSP:

        if (unlikely(len < L2CAP_CONF_RSP_SIZE(0))) {

            err = L2CAP_REJ_CMD_NOT_UNDERSTOOD;

            goto reject;

        }



        if (l2cap_channel_config_rsp_msg(l2cap,

                        le16_to_cpu(((l2cap_conf_rsp *) params)->result),

                        le16_to_cpu(((l2cap_conf_rsp *) params)->flags) & 1,

                        le16_to_cpu(((l2cap_conf_rsp *) params)->scid),

                        ((l2cap_conf_rsp *) params)->data,

                        len - L2CAP_CONF_RSP_SIZE(0)))

            fprintf(stderr, "%s: unexpected Configure Response (%02x) "

                            "packet, ignoring.\n", __func__, id);

        break;



    case L2CAP_DISCONN_REQ:

        if (unlikely(len != L2CAP_DISCONN_REQ_SIZE)) {

            err = L2CAP_REJ_CMD_NOT_UNDERSTOOD;

            goto reject;

        }



        l2cap_channel_close(l2cap,

                        le16_to_cpu(((l2cap_disconn_req *) params)->dcid),

                        le16_to_cpu(((l2cap_disconn_req *) params)->scid));

        break;



    case L2CAP_DISCONN_RSP:

        if (unlikely(len != L2CAP_DISCONN_RSP_SIZE)) {

            err = L2CAP_REJ_CMD_NOT_UNDERSTOOD;

            goto reject;

        }



        /* We never issue Disconnection Requests currently. TODO  */

        fprintf(stderr, "%s: unexpected Disconnection Response (%02x) "

                        "packet, ignoring.\n", __func__, id);

        break;



    case L2CAP_ECHO_REQ:

        l2cap_echo_response(l2cap, params, len);

        break;



    case L2CAP_ECHO_RSP:

        /* We never issue Echo Requests currently. TODO  */

        fprintf(stderr, "%s: unexpected Echo Response (%02x) "

                        "packet, ignoring.\n", __func__, id);

        break;



    case L2CAP_INFO_REQ:

        if (unlikely(len != L2CAP_INFO_REQ_SIZE)) {

            err = L2CAP_REJ_CMD_NOT_UNDERSTOOD;

            goto reject;

        }



        l2cap_info(l2cap, le16_to_cpu(((l2cap_info_req *) params)->type));

        break;



    case L2CAP_INFO_RSP:

        if (unlikely(len != L2CAP_INFO_RSP_SIZE)) {

            err = L2CAP_REJ_CMD_NOT_UNDERSTOOD;

            goto reject;

        }



        /* We never issue Information Requests currently. TODO  */

        fprintf(stderr, "%s: unexpected Information Response (%02x) "

                        "packet, ignoring.\n", __func__, id);

        break;



    default:

        err = L2CAP_REJ_CMD_NOT_UNDERSTOOD;

    reject:

        l2cap_command_reject(l2cap, id, err, 0, 0);

        break;

    }

}
