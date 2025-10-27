void qtest_quit(QTestState *s)

{

    int status;



    pid_t pid = qtest_qemu_pid(s);

    if (pid != -1) {

        kill(pid, SIGTERM);

        waitpid(pid, &status, 0);

    }






    unlink(s->pid_file);

    unlink(s->socket_path);

    unlink(s->qmp_socket_path);

    g_free(s->pid_file);

    g_free(s->socket_path);

    g_free(s->qmp_socket_path);


}