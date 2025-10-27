static int nbd_opt_go(QIOChannel *ioc, const char *wantname,

                      NBDExportInfo *info, Error **errp)

{

    nbd_opt_reply reply;

    uint32_t len = strlen(wantname);

    uint16_t type;

    int error;

    char *buf;



    /* The protocol requires that the server send NBD_INFO_EXPORT with

     * a non-zero flags (at least NBD_FLAG_HAS_FLAGS must be set); so

     * flags still 0 is a witness of a broken server. */

    info->flags = 0;



    trace_nbd_opt_go_start(wantname);

    buf = g_malloc(4 + len + 2 + 2 * info->request_sizes + 1);

    stl_be_p(buf, len);

    memcpy(buf + 4, wantname, len);

    /* At most one request, everything else up to server */

    stw_be_p(buf + 4 + len, info->request_sizes);

    if (info->request_sizes) {

        stw_be_p(buf + 4 + len + 2, NBD_INFO_BLOCK_SIZE);

    }

    if (nbd_send_option_request(ioc, NBD_OPT_GO,

                                4 + len + 2 + 2 * info->request_sizes, buf,

                                errp) < 0) {

        return -1;

    }



    while (1) {

        if (nbd_receive_option_reply(ioc, NBD_OPT_GO, &reply, errp) < 0) {

            return -1;

        }

        error = nbd_handle_reply_err(ioc, &reply, errp);

        if (error <= 0) {

            return error;

        }

        len = reply.length;



        if (reply.type == NBD_REP_ACK) {

            /* Server is done sending info and moved into transmission

               phase, but make sure it sent flags */

            if (len) {

                error_setg(errp, "server sent invalid NBD_REP_ACK");

                nbd_send_opt_abort(ioc);

                return -1;

            }

            if (!info->flags) {

                error_setg(errp, "broken server omitted NBD_INFO_EXPORT");

                nbd_send_opt_abort(ioc);

                return -1;

            }

            trace_nbd_opt_go_success();

            return 1;

        }

        if (reply.type != NBD_REP_INFO) {

            error_setg(errp, "unexpected reply type %" PRIx32

                       " (%s), expected %x",

                       reply.type, nbd_rep_lookup(reply.type), NBD_REP_INFO);

            nbd_send_opt_abort(ioc);

            return -1;

        }

        if (len < sizeof(type)) {

            error_setg(errp, "NBD_REP_INFO length %" PRIu32 " is too short",

                       len);

            nbd_send_opt_abort(ioc);

            return -1;

        }

        if (nbd_read(ioc, &type, sizeof(type), errp) < 0) {

            error_prepend(errp, "failed to read info type");

            nbd_send_opt_abort(ioc);

            return -1;

        }

        len -= sizeof(type);

        be16_to_cpus(&type);

        switch (type) {

        case NBD_INFO_EXPORT:

            if (len != sizeof(info->size) + sizeof(info->flags)) {

                error_setg(errp, "remaining export info len %" PRIu32

                           " is unexpected size", len);

                nbd_send_opt_abort(ioc);

                return -1;

            }

            if (nbd_read(ioc, &info->size, sizeof(info->size), errp) < 0) {

                error_prepend(errp, "failed to read info size");

                nbd_send_opt_abort(ioc);

                return -1;

            }

            be64_to_cpus(&info->size);

            if (nbd_read(ioc, &info->flags, sizeof(info->flags), errp) < 0) {

                error_prepend(errp, "failed to read info flags");

                nbd_send_opt_abort(ioc);

                return -1;

            }

            be16_to_cpus(&info->flags);

            trace_nbd_receive_negotiate_size_flags(info->size, info->flags);

            break;



        case NBD_INFO_BLOCK_SIZE:

            if (len != sizeof(info->min_block) * 3) {

                error_setg(errp, "remaining export info len %" PRIu32

                           " is unexpected size", len);

                nbd_send_opt_abort(ioc);

                return -1;

            }

            if (nbd_read(ioc, &info->min_block, sizeof(info->min_block),

                         errp) < 0) {

                error_prepend(errp, "failed to read info minimum block size");

                nbd_send_opt_abort(ioc);

                return -1;

            }

            be32_to_cpus(&info->min_block);

            if (!is_power_of_2(info->min_block)) {

                error_setg(errp, "server minimum block size %" PRId32

                           "is not a power of two", info->min_block);

                nbd_send_opt_abort(ioc);

                return -1;

            }

            if (nbd_read(ioc, &info->opt_block, sizeof(info->opt_block),

                         errp) < 0) {

                error_prepend(errp, "failed to read info preferred block size");

                nbd_send_opt_abort(ioc);

                return -1;

            }

            be32_to_cpus(&info->opt_block);

            if (!is_power_of_2(info->opt_block) ||

                info->opt_block < info->min_block) {

                error_setg(errp, "server preferred block size %" PRId32

                           "is not valid", info->opt_block);

                nbd_send_opt_abort(ioc);

                return -1;

            }

            if (nbd_read(ioc, &info->max_block, sizeof(info->max_block),

                         errp) < 0) {

                error_prepend(errp, "failed to read info maximum block size");

                nbd_send_opt_abort(ioc);

                return -1;

            }

            be32_to_cpus(&info->max_block);

            trace_nbd_opt_go_info_block_size(info->min_block, info->opt_block,

                                             info->max_block);

            break;



        default:

            trace_nbd_opt_go_info_unknown(type, nbd_info_lookup(type));

            if (nbd_drop(ioc, len, errp) < 0) {

                error_prepend(errp, "Failed to read info payload");

                nbd_send_opt_abort(ioc);

                return -1;

            }

            break;

        }

    }

}
