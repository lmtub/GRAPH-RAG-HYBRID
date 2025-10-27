static void curl_multi_check_completion(BDRVCURLState *s)

{

    int msgs_in_queue;



    /* Try to find done transfers, so we can free the easy

     * handle again. */

    do {

        CURLMsg *msg;

        msg = curl_multi_info_read(s->multi, &msgs_in_queue);



        if (!msg)

            break;

        if (msg->msg == CURLMSG_NONE)

            break;



        switch (msg->msg) {

            case CURLMSG_DONE:

            {

                CURLState *state = NULL;

                curl_easy_getinfo(msg->easy_handle, CURLINFO_PRIVATE,

                                  (char **)&state);



                /* ACBs for successful messages get completed in curl_read_cb */

                if (msg->data.result != CURLE_OK) {

                    int i;

                    for (i = 0; i < CURL_NUM_ACB; i++) {

                        CURLAIOCB *acb = state->acb[i];



                        if (acb == NULL) {

                            continue;

                        }



                        acb->common.cb(acb->common.opaque, -EIO);

                        qemu_aio_release(acb);

                        state->acb[i] = NULL;

                    }

                }



                curl_clean_state(state);

                break;

            }

            default:

                msgs_in_queue = 0;

                break;

        }

    } while(msgs_in_queue);

}
