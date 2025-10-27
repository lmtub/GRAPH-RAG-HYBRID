static void bt_l2cap_sdp_sdu_in(void *opaque, const uint8_t *data, int len)

{

    struct bt_l2cap_sdp_state_s *sdp = opaque;

    enum bt_sdp_cmd pdu_id;

    uint8_t rsp[MAX_PDU_OUT_SIZE - PDU_HEADER_SIZE], *sdu_out;

    int transaction_id, plen;

    int err = 0;

    int rsp_len = 0;



    if (len < 5) {

        fprintf(stderr, "%s: short SDP PDU (%iB).\n", __func__, len);

        return;

    }



    pdu_id = *data ++;

    transaction_id = (data[0] << 8) | data[1];

    plen = (data[2] << 8) | data[3];

    data += 4;

    len -= 5;



    if (len != plen) {

        fprintf(stderr, "%s: wrong SDP PDU length (%iB != %iB).\n",

                        __func__, plen, len);

        err = SDP_INVALID_PDU_SIZE;

        goto respond;

    }



    switch (pdu_id) {

    case SDP_SVC_SEARCH_REQ:

        rsp_len = sdp_svc_search(sdp, rsp, data, len);

        pdu_id = SDP_SVC_SEARCH_RSP;

        break;



    case SDP_SVC_ATTR_REQ:

        rsp_len = sdp_attr_get(sdp, rsp, data, len);

        pdu_id = SDP_SVC_ATTR_RSP;

        break;



    case SDP_SVC_SEARCH_ATTR_REQ:

        rsp_len = sdp_svc_search_attr_get(sdp, rsp, data, len);

        pdu_id = SDP_SVC_SEARCH_ATTR_RSP;

        break;



    case SDP_ERROR_RSP:

    case SDP_SVC_ATTR_RSP:

    case SDP_SVC_SEARCH_RSP:

    case SDP_SVC_SEARCH_ATTR_RSP:

    default:

        fprintf(stderr, "%s: unexpected SDP PDU ID %02x.\n",

                        __func__, pdu_id);

        err = SDP_INVALID_SYNTAX;

        break;

    }



    if (rsp_len < 0) {

        err = -rsp_len;

        rsp_len = 0;

    }



respond:

    if (err) {

        pdu_id = SDP_ERROR_RSP;

        rsp[rsp_len ++] = err >> 8;

        rsp[rsp_len ++] = err & 0xff;

    }



    sdu_out = sdp->channel->sdu_out(sdp->channel, rsp_len + PDU_HEADER_SIZE);



    sdu_out[0] = pdu_id;

    sdu_out[1] = transaction_id >> 8;

    sdu_out[2] = transaction_id & 0xff;

    sdu_out[3] = rsp_len >> 8;

    sdu_out[4] = rsp_len & 0xff;

    memcpy(sdu_out + PDU_HEADER_SIZE, rsp, rsp_len);



    sdp->channel->sdu_submit(sdp->channel);

}
