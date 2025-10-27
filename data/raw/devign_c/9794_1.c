static ssize_t sdp_svc_search(struct bt_l2cap_sdp_state_s *sdp,

                uint8_t *rsp, const uint8_t *req, ssize_t len)

{

    ssize_t seqlen;

    int i, count, start, end, max;

    int32_t handle;



    /* Perform the search */

    for (i = 0; i < sdp->services; i ++)

        sdp->service_list[i].match = 0;



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

    len = 4;

    count = 0;

    end = start;

    for (i = 0; i < sdp->services; i ++)

        if (sdp->service_list[i].match) {

            if (count >= start && count < max && len + 4 < MAX_RSP_PARAM_SIZE) {

                handle = i;

                memcpy(rsp + len, &handle, 4);

                len += 4;

                end = count + 1;

            }



            count ++;

        }



    rsp[0] = count >> 8;

    rsp[1] = count & 0xff;

    rsp[2] = (end - start) >> 8;

    rsp[3] = (end - start) & 0xff;



    if (end < count) {

        rsp[len ++] = sizeof(int);

        memcpy(rsp + len, &end, sizeof(int));

        len += 4;

    } else

        rsp[len ++] = 0;



    return len;

}
