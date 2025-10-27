QTestState *qtest_init_without_qmp_handshake(const char *extra_args)
{
    QTestState *s;
    int sock, qmpsock, i;
    gchar *socket_path;
    gchar *qmp_socket_path;
    gchar *command;
    const char *qemu_binary;
    qemu_binary = getenv("QTEST_QEMU_BINARY");
    g_assert(qemu_binary != NULL);
    s = g_malloc(sizeof(*s));
    socket_path = g_strdup_printf("/tmp/qtest-%d.sock", getpid());
    qmp_socket_path = g_strdup_printf("/tmp/qtest-%d.qmp", getpid());
    sock = init_socket(socket_path);
    qmpsock = init_socket(qmp_socket_path);
    qtest_add_abrt_handler(kill_qemu_hook_func, s);
    s->qemu_pid = fork();
    if (s->qemu_pid == 0) {
        setenv("QEMU_AUDIO_DRV", "none", true);
        command = g_strdup_printf("exec %s "
                                  "-qtest unix:%s,nowait "
                                  "-qtest-log %s "
                                  "-qmp unix:%s,nowait "
                                  "-machine accel=qtest "
                                  "-display none "
                                  "%s", qemu_binary, socket_path,
                                  getenv("QTEST_LOG") ? "/dev/fd/2" : "/dev/null",
                                  qmp_socket_path,
                                  extra_args ?: "");
        execlp("/bin/sh", "sh", "-c", command, NULL);
        exit(1);
    }
    s->fd = socket_accept(sock);
    if (s->fd >= 0) {
        s->qmp_fd = socket_accept(qmpsock);
    }
    g_free(socket_path);
    g_free(qmp_socket_path);
    g_assert(s->fd >= 0 && s->qmp_fd >= 0);
    s->rx = g_string_new("");
    for (i = 0; i < MAX_IRQ; i++) {
        s->irq_level[i] = false;
    }
    if (getenv("QTEST_STOP")) {
        kill(s->qemu_pid, SIGSTOP);
    }
    /* ask endianness of the target */
    s->big_endian = qtest_query_target_endianness(s);
    return s;
}