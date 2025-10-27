static ssize_t sdp_attr_get(struct bt_l2cap_sdp_state_s *sdp,

                uint8_t *rsp, const uint8_t *req, ssize_t len)

{

    ssize_t seqlen;

    int i, start, end, max;

    int32_t handle;

    struct sdp_service_record_s *record;

    uint8_t *lst;



    /* Perform the search */

    if (len < 7)

        return -SDP_INVALID_SYNTAX;

    memcpy(&handle, req, 4);

    req += 4;

    len -= 4;



    if (handle < 0 || handle > sdp->services)

        return -SDP_INVALID_RECORD_HANDLE;

    record = &sdp->service_list[handle];



    for (i = 0; i < record->attributes; i ++)

        record->attribute_list[i].match = 0;



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

            if (sdp_attr_match(record, &req, &seqlen))

                return -SDP_INVALID_SYNTAX;

    } else if (sdp_attr_match(record, &req, &seqlen))

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

    lst = rsp + 2;

    max = MIN(max, MAX_RSP_PARAM_SIZE);

    len = 3 - start;

    end = 0;

    for (i = 0; i < record->attributes; i ++)

        if (record->attribute_list[i].match) {

            if (len >= 0 && len + record->attribute_list[i].len < max) {

                memcpy(lst + len, record->attribute_list[i].pair,

                                record->attribute_list[i].len);

                end = len + record->attribute_list[i].len;

            }

            len += record->attribute_list[i].len;

        }

    if (0 >= start) {

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
