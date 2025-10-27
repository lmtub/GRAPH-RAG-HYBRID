static void do_closefd(Monitor *mon, const QDict *qdict)

{

    const char *fdname = qdict_get_str(qdict, "fdname");

    mon_fd_t *monfd;



    LIST_FOREACH(monfd, &mon->fds, next) {

        if (strcmp(monfd->name, fdname) != 0) {

            continue;

        }



        LIST_REMOVE(monfd, next);

        close(monfd->fd);

        qemu_free(monfd->name);

        qemu_free(monfd);

        return;

    }



    monitor_printf(mon, "Failed to find file descriptor named %s\n",

                   fdname);

}
