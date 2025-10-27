int tcp_socket_outgoing_opts(QemuOpts *opts)

{

    Error *local_err = NULL;

    int fd = inet_connect_opts(opts, &local_err, NULL, NULL);

    if (local_err != NULL) {

        qerror_report_err(local_err);

        error_free(local_err);

    }



    return fd;

}
