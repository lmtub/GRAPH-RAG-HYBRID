static int colo_do_checkpoint_transaction(MigrationState *s)

{

    Error *local_err = NULL;



    colo_send_message(s->to_dst_file, COLO_MESSAGE_CHECKPOINT_REQUEST,

                      &local_err);

    if (local_err) {

        goto out;

    }



    colo_receive_check_message(s->rp_state.from_dst_file,

                    COLO_MESSAGE_CHECKPOINT_REPLY, &local_err);

    if (local_err) {

        goto out;

    }



    /* TODO: suspend and save vm state to colo buffer */



    colo_send_message(s->to_dst_file, COLO_MESSAGE_VMSTATE_SEND, &local_err);

    if (local_err) {

        goto out;

    }



    /* TODO: send vmstate to Secondary */



    colo_receive_check_message(s->rp_state.from_dst_file,

                       COLO_MESSAGE_VMSTATE_RECEIVED, &local_err);

    if (local_err) {

        goto out;

    }



    colo_receive_check_message(s->rp_state.from_dst_file,

                       COLO_MESSAGE_VMSTATE_LOADED, &local_err);

    if (local_err) {

        goto out;

    }



    /* TODO: resume Primary */



    return 0;

out:

    if (local_err) {

        error_report_err(local_err);

    }

    return -EINVAL;

}
