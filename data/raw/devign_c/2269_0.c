static void *source_return_path_thread(void *opaque)

{

    MigrationState *ms = opaque;

    QEMUFile *rp = ms->rp_state.from_dst_file;

    uint16_t header_len, header_type;

    const int max_len = 512;

    uint8_t buf[max_len];

    uint32_t tmp32, sibling_error;

    ram_addr_t start = 0; /* =0 to silence warning */

    size_t  len = 0, expected_len;

    int res;



    trace_source_return_path_thread_entry();

    while (!ms->rp_state.error && !qemu_file_get_error(rp) &&

           migration_is_setup_or_active(ms->state)) {

        trace_source_return_path_thread_loop_top();

        header_type = qemu_get_be16(rp);

        header_len = qemu_get_be16(rp);



        if (header_type >= MIG_RP_MSG_MAX ||

            header_type == MIG_RP_MSG_INVALID) {

            error_report("RP: Received invalid message 0x%04x length 0x%04x",

                    header_type, header_len);

            mark_source_rp_bad(ms);

            goto out;

        }



        if ((rp_cmd_args[header_type].len != -1 &&

            header_len != rp_cmd_args[header_type].len) ||

            header_len > max_len) {

            error_report("RP: Received '%s' message (0x%04x) with"

                    "incorrect length %d expecting %zu",

                    rp_cmd_args[header_type].name, header_type, header_len,

                    (size_t)rp_cmd_args[header_type].len);

            mark_source_rp_bad(ms);

            goto out;

        }



        /* We know we've got a valid header by this point */

        res = qemu_get_buffer(rp, buf, header_len);

        if (res != header_len) {

            error_report("RP: Failed reading data for message 0x%04x"

                         " read %d expected %d",

                         header_type, res, header_len);

            mark_source_rp_bad(ms);

            goto out;

        }



        /* OK, we have the message and the data */

        switch (header_type) {

        case MIG_RP_MSG_SHUT:

            sibling_error = be32_to_cpup((uint32_t *)buf);

            trace_source_return_path_thread_shut(sibling_error);

            if (sibling_error) {

                error_report("RP: Sibling indicated error %d", sibling_error);

                mark_source_rp_bad(ms);

            }

            /*

             * We'll let the main thread deal with closing the RP

             * we could do a shutdown(2) on it, but we're the only user

             * anyway, so there's nothing gained.

             */

            goto out;



        case MIG_RP_MSG_PONG:

            tmp32 = be32_to_cpup((uint32_t *)buf);

            trace_source_return_path_thread_pong(tmp32);

            break;



        case MIG_RP_MSG_REQ_PAGES:

            start = be64_to_cpup((uint64_t *)buf);

            len = be32_to_cpup((uint32_t *)(buf + 8));

            migrate_handle_rp_req_pages(ms, NULL, start, len);

            break;



        case MIG_RP_MSG_REQ_PAGES_ID:

            expected_len = 12 + 1; /* header + termination */



            if (header_len >= expected_len) {

                start = be64_to_cpup((uint64_t *)buf);

                len = be32_to_cpup((uint32_t *)(buf + 8));

                /* Now we expect an idstr */

                tmp32 = buf[12]; /* Length of the following idstr */

                buf[13 + tmp32] = '\0';

                expected_len += tmp32;

            }

            if (header_len != expected_len) {

                error_report("RP: Req_Page_id with length %d expecting %zd",

                        header_len, expected_len);

                mark_source_rp_bad(ms);

                goto out;

            }

            migrate_handle_rp_req_pages(ms, (char *)&buf[13], start, len);

            break;



        default:

            break;

        }

    }

    if (rp && qemu_file_get_error(rp)) {

        trace_source_return_path_thread_bad_end();

        mark_source_rp_bad(ms);

    }



    trace_source_return_path_thread_end();

out:

    ms->rp_state.from_dst_file = NULL;

    qemu_fclose(rp);

    return NULL;

}
