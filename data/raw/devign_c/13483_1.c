static ssize_t sdp_svc_search_attr_get(struct bt_l2cap_sdp_state_s *sdp,

                uint8_t *rsp, const uint8_t *req, ssize_t len)

{

    ssize_t seqlen;

    int i, j, start, end, max;

    struct sdp_service_record_s *record;

    uint8_t *lst;



    /* Perform the search */

    for (i = 0; i < sdp->services; i ++) {

        sdp->service_list[i].match = 0;

            for (j = 0; j < sdp->service_list[i].attributes; j ++)

                sdp->service_list[i].attribute_list[j].match = 0;

    }



    if (len < 1)

        return -SDP_INVALID_SYNTAX;

    if ((*req & ~SDP_DSIZE_MASK) == SDP_DTYPE_SEQ) {

        seqlen = sdp_datalen(&req, &len);

        if (seqlen < 3 || len < seqlen)

            return -SDP_INVALID_SYNTAX;

        len -= seqlen;



        while (seqlen)

            if (sdp_svc_match(sdp, &req, &seqlen))

                return -SDP_INVALID_SYNTAX;

    } else if (sdp_svc_match(sdp, &req, &seqlen))

        return -SDP_INVALID_SYNTAX;



    if (len < 3)

        return -SDP_INVALID_SYNTAX;

    max = (req[0] << 8) | req[1];

    req += 2;

    len -= 2;

    if (max < 0x0007)

        return -SDP_INVALID_SYNTAX;



    if ((*req & ~SDP_DSIZE_MASK) == SDP_DTYPE_SEQ) {

        seqlen = sdp_datalen(&req, &len);

        if (seqlen < 3 || len < seqlen)

            return -SDP_INVALID_SYNTAX;

        len -= seqlen;



        while (seqlen)

            if (sdp_svc_attr_match(sdp, &req, &seqlen))

                return -SDP_INVALID_SYNTAX;

    } else if (sdp_svc_attr_match(sdp, &req, &seqlen))

        return -SDP_INVALID_SYNTAX;



    if (len < 1)

        return -SDP_INVALID_SYNTAX;



    if (*req) {

        if (len <= sizeof(int))

            return -SDP_INVALID_SYNTAX;

        len -= sizeof(int);

        memcpy(&start, req + 1, sizeof(int));

    } else

        start = 0;



    if (len > 1)

        return -SDP_INVALID_SYNTAX;



    /* Output the results */

    /* This assumes empty attribute lists are never to be returned even

     * for matching Service Records.  In practice this shouldn't happen

     * as the requestor will usually include the always present

     * ServiceRecordHandle AttributeID in AttributeIDList.  */

    lst = rsp + 2;

    max = MIN(max, MAX_RSP_PARAM_SIZE);

    len = 3 - start;

    end = 0;

    for (i = 0; i < sdp->services; i ++)

        if ((record = &sdp->service_list[i])->match) {

            len += 3;

            seqlen = len;

            for (j = 0; j < record->attributes; j ++)

                if (record->attribute_list[j].match) {

                    if (len >= 0)

                        if (len + record->attribute_list[j].len < max) {

                            memcpy(lst + len, record->attribute_list[j].pair,

                                            record->attribute_list[j].len);

                            end = len + record->attribute_list[j].len;

                        }

                    len += record->attribute_list[j].len;

                }

            if (seqlen == len)

                len -= 3;

            else if (seqlen >= 3 && seqlen < max) {

                lst[seqlen - 3] = SDP_DTYPE_SEQ | SDP_DSIZE_NEXT2;

                lst[seqlen - 2] = (len - seqlen) >> 8;

                lst[seqlen - 1] = (len - seqlen) & 0xff;

            }

        }

    if (len == 3 - start)

        len -= 3;

    else if (0 >= start) {

       lst[0] = SDP_DTYPE_SEQ | SDP_DSIZE_NEXT2;

       lst[1] = (len + start - 3) >> 8;

       lst[2] = (len + start - 3) & 0xff;

    }



    rsp[0] = end >> 8;

    rsp[1] = end & 0xff;



    if (end < len) {

        len = end + start;

        lst[end ++] = sizeof(int);

        memcpy(lst + end, &len, sizeof(int));

        end += sizeof(int);

    } else

        lst[end ++] = 0;



    return end + 2;

}
